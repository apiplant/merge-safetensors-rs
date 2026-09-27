//! Core logic for merging sharded `.safetensors` files based on a
//! `*.safetensors.index.json` file, without ever holding more than one
//! chunk of tensor data in memory at a time.
//!
//! Strategy:
//! 1. Read only the small JSON headers of every shard (in parallel) to learn
//!    each tensor's dtype/shape/byte-range within its shard.
//! 2. Compute the merged header (global byte offsets) purely from that
//!    metadata, matching the ordering the `safetensors` crate itself uses
//!    (descending dtype alignment, then name) so the output is a byte-for-byte
//!    valid safetensors file.
//! 3. Write the merged header, then stream each tensor's bytes straight from
//!    its source shard file to the output file through a small fixed-size
//!    buffer — no shard is ever fully loaded into memory.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use safetensors::tensor::{Dtype, TensorInfo};
use serde::Deserialize;

/// Size of the fixed buffer used to stream tensor bytes from a shard to the
/// output file. Memory usage of the copy phase is bounded by this constant,
/// independent of tensor or shard size.
const COPY_BUFFER_SIZE: usize = 1 << 20; // 1 MiB

#[derive(Debug, Deserialize)]
struct IndexFile {
    #[serde(default)]
    metadata: Option<IndexMetadata>,
    weight_map: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct IndexMetadata {
    total_size: Option<u64>,
}

/// One shard's parsed header: where its tensor data begins in the file, and
/// the per-tensor byte ranges within that data section.
struct ShardHeader {
    data_start: u64,
    tensors: HashMap<String, TensorInfo>,
}

pub struct MergeOptions {
    pub index_file: PathBuf,
    pub output: PathBuf,
    pub verbose: bool,
    pub show_progress: bool,
}

pub struct MergeReport {
    pub num_tensors: usize,
    pub num_shards: usize,
    pub output_path: PathBuf,
    pub bytes_written: u64,
    pub load_duration: Duration,
    pub save_duration: Duration,
}

macro_rules! log_info {
    ($verbose_ctx:expr, $($arg:tt)*) => {
        println!("[INFO] {}", format!($($arg)*));
    };
}

macro_rules! log_debug {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!("[DEBUG] {}", format!($($arg)*));
        }
    };
}

fn read_shard_header(path: &Path) -> Result<ShardHeader> {
    let mut file = File::open(path)
        .with_context(|| format!("Shard file not found: {}", path.display()))?;

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .with_context(|| format!("Failed to read header size from {}", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf);

    let mut header_buf = vec![0u8; header_len as usize];
    file.read_exact(&mut header_buf)
        .with_context(|| format!("Failed to read header from {}", path.display()))?;

    let raw: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header_buf)
        .with_context(|| format!("Invalid safetensors header JSON in {}", path.display()))?;

    let mut tensors = HashMap::with_capacity(raw.len());
    for (key, value) in raw {
        if key == "__metadata__" {
            continue;
        }
        let info: TensorInfo = serde_json::from_value(value)
            .with_context(|| format!("Invalid tensor info for '{key}' in {}", path.display()))?;
        tensors.insert(key, info);
    }

    Ok(ShardHeader {
        data_start: 8 + header_len,
        tensors,
    })
}

/// One tensor resolved to its physical location, ready to be copied.
struct ResolvedTensor {
    name: String,
    shard_path: PathBuf,
    dtype: Dtype,
    shape: Vec<usize>,
    /// Absolute byte offset of this tensor's data within its shard file.
    abs_offset: u64,
    len: u64,
}

/// Builds the merged safetensors header bytes (matching the on-disk format
/// the reference `safetensors` crate produces: an 8-byte little-endian
/// header length, followed by the JSON header padded with spaces to a
/// multiple of 8 bytes) and returns it together with the tensors in the
/// exact order their bytes must be written in.
fn build_header(mut resolved: Vec<ResolvedTensor>) -> (Vec<u8>, Vec<ResolvedTensor>) {
    // Mirrors `safetensors::tensor::prepare`: sort by descending dtype
    // alignment, then ascending name, so re-merged files are consistent
    // with what the original crate would produce.
    resolved.sort_by(|a, b| b.dtype.cmp(&a.dtype).then_with(|| a.name.cmp(&b.name)));

    let mut header_map: BTreeMap<String, TensorInfo> = BTreeMap::new();
    let mut offset: u64 = 0;
    for t in &resolved {
        let info = TensorInfo {
            dtype: t.dtype,
            shape: t.shape.clone(),
            data_offsets: (offset as usize, (offset + t.len) as usize),
        };
        offset += t.len;
        header_map.insert(t.name.clone(), info);
    }

    let mut header_bytes = serde_json::to_vec(&header_map).expect("header map is always valid JSON");
    let padding = (8 - header_bytes.len() % 8) % 8;
    header_bytes.extend(std::iter::repeat(b' ').take(padding));

    (header_bytes, resolved)
}

fn copy_range(
    file: &mut File,
    abs_offset: u64,
    len: u64,
    writer: &mut impl Write,
    buf: &mut [u8],
    pb: &ProgressBar,
) -> Result<()> {
    file.seek(SeekFrom::Start(abs_offset))?;
    let mut remaining = len;
    while remaining > 0 {
        let chunk = remaining.min(buf.len() as u64) as usize;
        file.read_exact(&mut buf[..chunk])?;
        writer.write_all(&buf[..chunk])?;
        remaining -= chunk as u64;
        pb.inc(chunk as u64);
    }
    Ok(())
}

pub fn run(opts: MergeOptions) -> Result<MergeReport> {
    let start = Instant::now();

    let index_path = opts
        .index_file
        .canonicalize()
        .with_context(|| format!("Index file not found: {}", opts.index_file.display()))?;
    let index_dir = index_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    log_info!(opts.verbose, "Loading index file: {}", index_path.display());
    let index_data = std::fs::read_to_string(&index_path)
        .with_context(|| format!("Failed to read index file: {}", index_path.display()))?;
    let index: IndexFile = serde_json::from_str(&index_data)
        .with_context(|| format!("Invalid JSON in index file: {}", index_path.display()))?;

    if index.weight_map.is_empty() {
        bail!(
            "'weight_map' key not found or empty in index file: {}",
            index_path.display()
        );
    }

    // Group keys by shard file, resolved relative to the index's directory.
    let mut shards_to_load: HashMap<PathBuf, Vec<String>> = HashMap::new();
    for (key, shard_rel) in &index.weight_map {
        let shard_abs = index_dir.join(shard_rel);
        shards_to_load.entry(shard_abs).or_default().push(key.clone());
    }

    let mut shard_paths: Vec<PathBuf> = shards_to_load.keys().cloned().collect();
    shard_paths.sort();

    log_info!(opts.verbose, "Reading headers of {} shard(s) in parallel...", shard_paths.len());
    let load_start = Instant::now();

    let headers: HashMap<PathBuf, ShardHeader> = shard_paths
        .par_iter()
        .map(|path| -> Result<(PathBuf, ShardHeader)> {
            let header = read_shard_header(path)?;
            Ok((path.clone(), header))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .collect();

    let mut resolved = Vec::with_capacity(index.weight_map.len());
    for path in &shard_paths {
        let shard_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
        let header = &headers[path];
        for key in &shards_to_load[path] {
            match header.tensors.get(key) {
                Some(info) => {
                    log_debug!(opts.verbose, "  - Resolved tensor: {key} (in {shard_name})");
                    resolved.push(ResolvedTensor {
                        name: key.clone(),
                        shard_path: path.clone(),
                        dtype: info.dtype,
                        shape: info.shape.clone(),
                        abs_offset: header.data_start + info.data_offsets.0 as u64,
                        len: (info.data_offsets.1 - info.data_offsets.0) as u64,
                    });
                }
                None => {
                    eprintln!(
                        "[WARN] Key '{key}' specified in index not found in shard '{shard_name}'. Skipping."
                    );
                }
            }
        }
    }

    if resolved.is_empty() {
        bail!("No tensors were resolved. Check the index file, shard availability, and paths relative to the index.");
    }

    let total_bytes: u64 = resolved.iter().map(|t| t.len).sum();
    if let Some(meta) = &index.metadata {
        if let Some(expected) = meta.total_size {
            if expected != total_bytes {
                eprintln!(
                    "[WARN] Index metadata declares total_size={expected} bytes, but {total_bytes} bytes were found across resolved tensors. The index and shards may be mismatched."
                );
            } else {
                log_debug!(opts.verbose, "total_size from index matches resolved tensor bytes ({total_bytes}).");
            }
        }
    }

    log_info!(
        opts.verbose,
        "Resolved {} tensor(s) across {} shard(s) in {:.2}s.",
        resolved.len(),
        shard_paths.len(),
        load_start.elapsed().as_secs_f64()
    );

    // Build header and get tensors in final write order.
    let (header_bytes, ordered) = build_header(resolved);
    let header_len = header_bytes.len() as u64;

    if let Some(parent) = opts.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Could not create output directory: {}", parent.display()))?;
        }
    }
    if shard_paths.contains(&opts.output) {
        eprintln!(
            "[WARN] Output filename '{}' is the same as one of the input shards. This will overwrite the shard.",
            opts.output.display()
        );
    }

    log_info!(
        opts.verbose,
        "Preparing to save merged model to: {} (streaming, {} bytes of tensor data)...",
        opts.output.display(),
        total_bytes
    );
    let save_start = Instant::now();

    let out_file = File::create(&opts.output)
        .with_context(|| format!("Failed to create output file: {}", opts.output.display()))?;
    let mut writer = BufWriter::new(out_file);
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header_bytes)?;

    let pb = if opts.show_progress {
        let pb = ProgressBar::new(total_bytes);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        pb
    } else {
        ProgressBar::hidden()
    };

    // Keep one open file handle per shard (bounded by shard count, not data
    // size) so we don't repeatedly reopen shards that interleave in the
    // sorted write order.
    let mut open_files: HashMap<PathBuf, File> = HashMap::new();
    let mut buf = vec![0u8; COPY_BUFFER_SIZE];

    for tensor in &ordered {
        let file = match open_files.get_mut(&tensor.shard_path) {
            Some(f) => f,
            None => {
                let f = File::open(&tensor.shard_path).with_context(|| {
                    format!("Failed to reopen shard: {}", tensor.shard_path.display())
                })?;
                open_files.entry(tensor.shard_path.clone()).or_insert(f)
            }
        };
        copy_range(file, tensor.abs_offset, tensor.len, &mut writer, &mut buf, &pb)
            .with_context(|| format!("Failed to copy tensor '{}'", tensor.name))?;
    }
    writer.flush()?;
    pb.finish_and_clear();

    let save_duration = save_start.elapsed();
    log_info!(
        opts.verbose,
        "Successfully saved merged model in {:.2}s.",
        save_duration.as_secs_f64()
    );
    log_info!(
        opts.verbose,
        "Merge finished. Total time: {:.2}s.",
        start.elapsed().as_secs_f64()
    );

    Ok(MergeReport {
        num_tensors: ordered.len(),
        num_shards: shard_paths.len(),
        output_path: opts.output,
        bytes_written: header_len + 8 + total_bytes,
        load_duration: load_start.elapsed(),
        save_duration,
    })
}
