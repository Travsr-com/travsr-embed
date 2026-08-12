# travsr-embed

Embedding sidecar for [Travsr](https://github.com/Travsr-com/travsr). Encodes graph
nodes and symbols into vectors with the BGE model family (and, with the `ort`
feature, other ONNX model families) and maintains a disk-persistent usearch HNSW
index for nearest-neighbour retrieval.

The `travsr` CLI downloads a prebuilt `travsr-embed` binary per platform from this
repo's [GitHub Releases](https://github.com/Travsr-com/travsr-embed/releases); most
users never build it directly.

## Binaries

| Binary | Purpose |
| --- | --- |
| `travsr-embed` | The sidecar itself: reindexes a graph's nodes/symbols into `embed.db` and its HNSW index. Invoked by the `travsr` daemon, not typically run by hand. |
| `embed-jsonl` | Offline text-to-vector tool for retrieval prototyping. Reads JSONL `{"id", "text"}`, writes JSONL `{"id", "dim", "b64"}`. Touches no database or index, and shares the sidecar's engine cascade so vectors are byte-identical to production. |
| `bench-tract` | Benchmark harness for the `tract` backend. |

## Building

Requires Rust 1.91+ and a checkout of the sibling
[`travsr`](https://github.com/Travsr-com/travsr) repo one directory up (this crate
path-depends on `travsr-plugin-protocol` / `travsr-plugin-sdk` from it via
`[patch.crates-io]`):

```
cd ..
git clone git@github.com:Travsr-com/travsr.git
cd travsr-embed
cargo build --release
```

### Features

The default build compiles only the `tract` backend (pure Rust, zero extra
runtime deps). ONNX Runtime support is opt-in:

| Feature | Adds |
| --- | --- |
| `ort` | ONNX Runtime CPU execution, the universal fallback for model families `tract` cannot run (ModernBERT, nomic-bert, ...). |
| `ort-coreml` | Hardware acceleration on macOS (ANE + GPU), zero extra install. |
| `ort-cuda` | Hardware acceleration on Linux/Windows x86_64. Needs a host CUDA 12 runtime + cuDNN. |
| `ort-tensorrt` | TensorRT execution provider. Needs the same CUDA 12 host stack as `ort-cuda`. |

Backend selection at runtime is capability-based: each registered backend
declares which model families it can run, and the sidecar picks the first
backend, in preference order, that can serve the model actually being loaded.

## Usage

`travsr-embed` is normally launched by the `travsr` daemon, which passes it a
graph database to reindex:

```
travsr-embed --reindex <graph.db> --embed-db <embed.db> [--model-id <id>]
```

Common flags:

- `--model-id <id>` selects the embedding model at runtime (default is the
  BGE-small family).
- `--parallel <N>` reindexes with N worker threads (mutually exclusive with
  `--shard` / `--row-start`+`--row-end`). Pair with `--cancel-sentinel <path>`
  and `--rebuild-on-cancel` for graceful cancellation mid-reindex.
- `--shard <idx>/<total>` and `--row-start`/`--row-end` partition a reindex
  across external worker processes.
- `--rebuild-index <graph.db>` rebuilds the HNSW index from `embed.db` without
  re-embedding.
- `--phase1 <shell-threshold>` / `--phase2 <shell-threshold>` run only one
  phase of the two-phase symbol-level build.
- `--version` / `-V` prints the crate version and exits 0 (used by the
  `travsr` daemon as a capability probe).

Run with `--help`-shaped errors (any missing/invalid flag prints its own
`usage:` line and exits non-zero) or see `src/main.rs` for the full argument
matrix.

## Release process

Tagging a commit `vX.Y.Z` and pushing the tag triggers `.github/workflows/release.yml`,
which builds `travsr-embed` for macOS (arm64/x86_64), Linux (arm64/x86_64), and
Windows (x86_64), and publishes the binaries plus SHA256 sidecars as a GitHub
Release. See [CHANGELOG.md](CHANGELOG.md) for what shipped in each release.

## License

Apache-2.0. See [LICENSE](LICENSE).
