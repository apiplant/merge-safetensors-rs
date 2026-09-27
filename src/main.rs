use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use merge_safetensors::{run, MergeOptions};

#[derive(Parser, Debug)]
#[command(
    name = "merge-safetensors",
    about = "Merge sharded safetensors files into a single file based on an index.json",
    version
)]
struct Args {
    /// Path to the *.safetensors.index.json file
    #[arg(default_value = "model.safetensors.index.json")]
    index_file: PathBuf,

    /// Path for the merged output *.safetensors file (default: model-merged.safetensors)
    #[arg(default_value = "model-merged.safetensors")]
    output_file: PathBuf,

    /// Path for the merged output *.safetensors file (overrides the positional argument)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Enable verbose (debug) logging
    #[arg(short, long)]
    verbose: bool,

    /// Disable the progress bar (also disabled automatically when not attached to a terminal)
    #[arg(long)]
    no_progress: bool,
}

fn normalize_output_filename(mut name: PathBuf) -> PathBuf {
    let has_ext = name
        .extension()
        .map(|e| e.eq_ignore_ascii_case("safetensors"))
        .unwrap_or(false);
    if !has_ext {
        let mut s = name.into_os_string();
        s.push(".safetensors");
        name = PathBuf::from(s);
    }
    name
}

fn main() -> Result<()> {
    let args = Args::parse();

    let output = normalize_output_filename(args.output.unwrap_or(args.output_file));
    let show_progress = !args.no_progress && std::io::IsTerminal::is_terminal(&std::io::stderr());

    let report = run(MergeOptions {
        index_file: args.index_file,
        output,
        verbose: args.verbose,
        show_progress,
    })?;

    println!(
        "\nMerged {} tensor(s) from {} shard(s) into {} ({:.2} MiB) — load {:.2}s, save {:.2}s.",
        report.num_tensors,
        report.num_shards,
        report.output_path.display(),
        report.bytes_written as f64 / (1024.0 * 1024.0),
        report.load_duration.as_secs_f64(),
        report.save_duration.as_secs_f64(),
    );

    Ok(())
}
