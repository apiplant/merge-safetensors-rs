# merge-safetensors

A small Rust CLI that merges sharded `.safetensors` files (as produced by
`transformers`/`safetensors`-based checkpoints) into a single file, driven by
a `*.safetensors.index.json`.

## Why streaming instead of loading everything into memory?

A naive merge approach loads every tensor into memory for the whole run,
then copies it all again on save. This tool never holds more than one
shard's *header* and one fixed-size (1 MiB) buffer in memory — tensor bytes
are streamed directly from each source shard file to the output file.

## How it works

1. Read every shard's small JSON header (in parallel, via `rayon`) to learn
   each tensor's dtype, shape, and byte range within its shard — without
   touching any tensor data.
2. Compute the merged file's header entirely from that metadata, using the
   same ordering (descending dtype alignment, then tensor name) as the
   reference `safetensors` crate, so the output is byte-identical in
   structure to what `safetensors::serialize` would produce.
3. Write the merged header, then stream each tensor's bytes straight from its
   source shard into the output file through a fixed 1 MiB buffer.

Peak memory usage is therefore bounded by the buffer size and the number of
shards' file handles — independent of model size.

## Usage

```sh
cargo build --release

./target/release/merge-safetensors model.safetensors.index.json merged.safetensors
```

```
Usage: merge-safetensors [OPTIONS] [INDEX_FILE] [OUTPUT_FILE]

Arguments:
  [INDEX_FILE]   Path to the *.safetensors.index.json file [default: model.safetensors.index.json]
  [OUTPUT_FILE]  Path for the merged output *.safetensors file [default: model-merged.safetensors]

Options:
  -o, --output <OUTPUT>  Path for the merged output *.safetensors file (overrides the positional argument)
  -v, --verbose          Enable verbose (debug) logging
      --no-progress      Disable the progress bar (also disabled automatically when not attached to a terminal)
  -h, --help             Print help
  -V, --version          Print version
```

If the index file's `metadata.total_size` field is present, it's checked
against the sum of resolved tensor bytes and a warning is printed on
mismatch (a cheap sanity check the index and shards actually correspond to
each other).

## Development

```sh
cargo test      # integration tests build synthetic shards in-process, no Python needed
cargo build --release
```

## Install

Prebuilt packages (merge-safetensors) for macOS (Apple Silicon), Linux x86_64 and Linux arm64:

```bash
brew tap apiplant/tap && brew install apiplant/tap/merge-safetensors-rs      # macOS, Linux
sudo apt install merge-safetensors-rs      # Debian/Ubuntu, after adding apt.apiplant.com
sudo pacman -S merge-safetensors-rs        # Arch, after adding apiplant.github.io/pacman
```

Setup commands for the apt and pacman repositories, the plain archives and the release process are in [`packaging/README.md`](packaging/README.md). Release archives are on the [releases page](https://github.com/apiplant/merge-safetensors-rs/releases).
