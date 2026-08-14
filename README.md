# travsr-embed

Embedding sidecar for [Travsr](https://github.com/Travsr-com/travsr). Encodes graph
nodes and symbols into vectors with the BGE model family (and, with the `ort`
feature, other ONNX model families) and maintains a disk-persistent usearch HNSW
index for nearest-neighbour retrieval.

The `travsr` CLI downloads a prebuilt `travsr-embed` binary per platform from this
repo's [GitHub Releases](https://github.com/Travsr-com/travsr-embed/releases); most
users never build it directly.

## Which release asset do I want?

The default asset for your platform always works. GPU assets are opt-in **except
on macOS**, where acceleration is free (statically linked, nothing to install):

| Your machine | Asset | Accelerator | Host prerequisite |
| --- | --- | --- | --- |
| macOS (Apple Silicon or Intel) | default | CoreML — ANE + GPU | none |
| Linux x86_64 + NVIDIA GPU | `…-x86_64-unknown-linux-gnu-cuda` | CUDA | CUDA runtime + cuDNN, and a Haswell-or-newer CPU (`x86-64-v3`) |
| Linux x86_64, no GPU | default | — (tract, CPU) | none |
| Linux aarch64 (incl. OCI) | default | — (tract, CPU) | none |
| Windows x86_64 | default | — (tract, CPU) | none |

The `-cuda` asset ships extra `libonnxruntime_providers_*.so` files. **Keep them
in the same directory as the binary** — it finds them via an `$ORIGIN` rpath. Each
shipped file has a `.sha256` sidecar.

Nothing silently degrades: if a GPU asset cannot initialize its accelerator (no
driver, wrong CUDA version, missing provider libraries), the sidecar logs the
reason and falls back to CPU inference rather than failing. The worst case of
downloading the wrong asset is CPU speed, not a broken install.

Not yet available, and why:

- **Windows + NVIDIA** — ONNX Runtime publishes a CUDA build for
  `x86_64-pc-windows-msvc`, but it expects the *dynamic* CRT, while this repo
  forces the static one (`/MT`, required by `esaxx-rs`). Linking fails on
  `__imp_*` symbols. Not a platform gap and not unknown — CI probes it on every
  PR. The fix under consideration is ORT's `load-dynamic` mode, which loads
  `onnxruntime.dll` at run time and so avoids the CRT conflict entirely.
- **AMD / Intel GPUs** — the providers that drive them natively (DirectML, ROCm,
  and OpenVINO, which is also the only route to an Intel NPU) have no prebuilt
  ONNX Runtime for any target here, so reaching them means building ONNX Runtime
  from source. The portable alternative is WebGPU, which runs through D3D12 /
  Vulkan / Metal and so works on any modern adapter: `ort-webgpu` builds and is
  CI-checked **on Linux**, and upstream still marks it experimental, so it is not
  published yet.
- **Windows GPU** — works via `ort-directml`, which drives **any DX12 adapter**:
  Intel, AMD and NVIDIA alike. It is built differently from the other GPU
  features: no ORT is linked at build time (that is impossible on Windows — the
  static CRT this repo needs for `esaxx-rs` conflicts with ORT's prebuilt, see
  CONTRIBUTING), so `onnxruntime.dll` is loaded at run time instead, from beside
  the executable or from `ORT_DYLIB_PATH`.

  The DLL must be a DirectML-enabled build of ONNX Runtime. If it is missing the
  sidecar logs the reason and runs on CPU — it does not fail, and it does not
  hang.

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
| `ort-cuda` | Hardware acceleration on Linux/Windows x86_64. Needs a host CUDA runtime + cuDNN. |
| `ort-tensorrt` | TensorRT execution provider. Needs the same CUDA host stack as `ort-cuda`. |
| `ort-webgpu` | Vendor-neutral GPU (AMD/Intel/NVIDIA) via DX12/Vulkan/Metal. **Experimental upstream**; not shipped as a release asset yet. |

Only providers with a prebuilt ONNX Runtime are listed. `ort` exposes DirectML,
ROCm, OpenVINO and XNNPACK features too, but none has a prebuilt for a target this
repo ships, so using them would mean building ONNX Runtime from source. When
prebuilts appear, each is one line in the execution-provider cascade plus a
feature — see `src/backend/ort.rs`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for which of these can be built on which
host (notably: no ORT feature builds on a Windows *GNU* toolchain).

Backend selection at runtime is capability-based: each registered backend
declares which model families it can run, and the sidecar picks the first
backend, in preference order, that can serve the model actually being loaded.

### Reporting what a build can do

```bash
travsr-embed --capabilities
```

Prints the compiled engines, the model families they can run, and whether an
accelerator is compiled in, as JSON. The `travsr` CLI uses this to refuse a model
the installed sidecar cannot execute at *selection* time, instead of failing part
way through a background reindex. `accelerated_compiled` reports what was
compiled, not whether this machine's GPU will actually be confirmed at load time —
that answer needs a real model load, and appears in the startup log.

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
