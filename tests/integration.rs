use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, SafeTensors};
use serde::Serialize;

/// Writes a minimal, spec-correct safetensors file by hand (no torch/numpy
/// involved) so the test suite has no external dependency.
fn write_shard(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
    #[derive(Serialize)]
    struct Info {
        dtype: Dtype,
        shape: Vec<usize>,
        data_offsets: (usize, usize),
    }

    let mut header: std::collections::BTreeMap<String, Info> = std::collections::BTreeMap::new();
    let mut offset = 0usize;
    let mut blob = Vec::new();
    for (name, dtype, shape, data) in tensors {
        header.insert(
            name.to_string(),
            Info {
                dtype: *dtype,
                shape: shape.clone(),
                data_offsets: (offset, offset + data.len()),
            },
        );
        offset += data.len();
        blob.extend_from_slice(data);
    }

    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    let pad = (8 - header_bytes.len() % 8) % 8;
    header_bytes.extend(std::iter::repeat(b' ').take(pad));

    let mut f = File::create(path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&header_bytes).unwrap();
    f.write_all(&blob).unwrap();
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i64_bytes(vals: &[i64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn merges_shards_across_dtypes_and_preserves_bytes() {
    let dir = tempfile::tempdir().unwrap();

    let w1 = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let w2 = f32_bytes(&[5.0, 6.0]);
    let ids = i64_bytes(&[7, 8, 9]);

    write_shard(
        &dir.path().join("shard1.safetensors"),
        &[
            ("layer.weight", Dtype::F32, vec![2, 2], w1.clone()),
            ("layer.bias", Dtype::F32, vec![2], w2.clone()),
        ],
    );
    write_shard(
        &dir.path().join("shard2.safetensors"),
        &[("token_ids", Dtype::I64, vec![3], ids.clone())],
    );

    let mut weight_map = HashMap::new();
    weight_map.insert("layer.weight".to_string(), "shard1.safetensors".to_string());
    weight_map.insert("layer.bias".to_string(), "shard1.safetensors".to_string());
    weight_map.insert("token_ids".to_string(), "shard2.safetensors".to_string());

    let index = serde_json::json!({
        "metadata": { "total_size": w1.len() + w2.len() + ids.len() },
        "weight_map": weight_map,
    });
    let index_path = dir.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

    let output_path: PathBuf = dir.path().join("merged.safetensors");
    let report = merge_safetensors::run(merge_safetensors::MergeOptions {
        index_file: index_path,
        output: output_path.clone(),
        verbose: false,
        show_progress: false,
    })
    .expect("merge should succeed");

    assert_eq!(report.num_tensors, 3);
    assert_eq!(report.num_shards, 2);

    let merged_bytes = std::fs::read(&output_path).unwrap();
    let tensors = SafeTensors::deserialize(&merged_bytes).unwrap();

    let a = tensors.tensor("layer.weight").unwrap();
    assert_eq!(a.shape(), &[2, 2]);
    assert_eq!(a.dtype(), Dtype::F32);
    assert_eq!(a.data(), w1.as_slice());

    let b = tensors.tensor("layer.bias").unwrap();
    assert_eq!(b.shape(), &[2]);
    assert_eq!(b.data(), w2.as_slice());

    let c = tensors.tensor("token_ids").unwrap();
    assert_eq!(c.shape(), &[3]);
    assert_eq!(c.dtype(), Dtype::I64);
    assert_eq!(c.data(), ids.as_slice());
}

#[test]
fn skips_missing_keys_without_failing() {
    let dir = tempfile::tempdir().unwrap();
    let data = f32_bytes(&[1.0, 2.0]);
    write_shard(
        &dir.path().join("shard1.safetensors"),
        &[("present", Dtype::F32, vec![2], data.clone())],
    );

    let mut weight_map = HashMap::new();
    weight_map.insert("present".to_string(), "shard1.safetensors".to_string());
    weight_map.insert("ghost".to_string(), "shard1.safetensors".to_string());

    let index = serde_json::json!({ "weight_map": weight_map });
    let index_path = dir.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

    let output_path = dir.path().join("merged.safetensors");
    let report = merge_safetensors::run(merge_safetensors::MergeOptions {
        index_file: index_path,
        output: output_path.clone(),
        verbose: false,
        show_progress: false,
    })
    .expect("merge should succeed even with one missing key");

    assert_eq!(report.num_tensors, 1);
    let merged_bytes = std::fs::read(&output_path).unwrap();
    let tensors = SafeTensors::deserialize(&merged_bytes).unwrap();
    assert!(tensors.tensor("present").is_ok());
    assert!(tensors.tensor("ghost").is_err());
}

#[test]
fn fails_on_missing_index_file() {
    let dir = tempfile::tempdir().unwrap();
    let result = merge_safetensors::run(merge_safetensors::MergeOptions {
        index_file: dir.path().join("does-not-exist.json"),
        output: dir.path().join("out.safetensors"),
        verbose: false,
        show_progress: false,
    });
    assert!(result.is_err());
}
