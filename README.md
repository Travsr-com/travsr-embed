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
| Linux x86_64 + NVIDIA GPU | `…-x86_64-unknown-linux-gnu-cuda` | CUDA | CUDA runtime + cuDNN, glibc 2.38+, and a Haswell-or-newer CPU (`x86-64-v3`) |
| Windows x86_64 + any GPU | `…-x86_64-pc-windows-msvc-directml.exe` | DirectML — Intel, AMD **or** NVIDIA | a DirectX 12 GPU |
| Linux x86_64, no GPU | default | — (tract, CPU) | none |
| Linux aarch64 (incl. OCI) | default | — (tract, CPU) | none |
| Windows x86_64, no GPU | default | — (tract, CPU) | none |

Accelerated assets ship their ONNX Runtime libraries alongside the binary
(`libonnxruntime_providers_*.so` for `-cuda`, `onnxruntime.dll` for `-directml`).
**Keep them in the same directory as the binary** — `-cuda` finds them via an
`$ORIGIN` rpath, and `-directml` loads its DLL at run time from beside the
executable (or from `ORT_DYLIB_PATH`). Each shipped file has a `.sha256` sidecar.

The glibc 2.38 floor on `-cuda` is inherited from ORT's prebuilt, which
references `__isoc23_*` symbols added in that release. It rules out Ubuntu 22.04,
Debian 12 and RHEL 9 — use the default CPU asset there, which has no such floor.

Nothing silently degrades: if a GPU asset cannot initialize its accelerator (no
driver, wrong CUDA version, missing provider libraries), the sidecar logs the
reason and falls back to CPU inference rather than failing. The worst case of
downloading the wrong asset is CPU speed, not a broken install.

Windows deserves a note, because it is built differently from every other GPU
asset. No ONNX Runtime is linked into it at all: the static CRT this repo needs
for `esaxx-rs` conflicts with ORT's prebuilt, and that blocks *every* statically
linked ORT feature there — CPU-only included, since the unresolved `__imp_*`
symbols come from ONNX Runtime's own object code. The `-directml` asset sidesteps
that by loading `onnxruntime.dll` at run time instead, which is also what frees it
from ORT's prebuilt set and makes DirectML reachable in the first place.

Not yet available, and why:

- **Intel NPU** (and any other OpenVINO-only target) — OpenVINO is the sole route
  to an NPU, and it has no prebuilt ONNX Runtime for any target here. Reaching it
  means building ONNX Runtime from source. Now that Windows loads its runtime
  dynamically the constraint is softer than it looks — an OpenVINO-enabled DLL
  would be the same shape of solution as the DirectML one — but nobody has tried
  it.
- **ROCm** — same story: a Cargo feature exists upstream, no prebuilt does.
- **WebGPU** — the portable path (D3D12 / Vulkan / Metal, so any modern adapter).
  `ort-webgpu` builds, and CI compiles it on Linux and Windows, though those
  checks are advisory (`continue-on-error`) rather than blocking. Upstream still
  marks WebGPU experimental, so no asset is published. On Windows it is redundant
  with DirectML anyway.

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
| `ort-cuda` | Hardware acceleration on Linux x86_64. Needs a host CUDA runtime + cuDNN and glibc 2.38+. |
| `ort-tensorrt` | TensorRT execution provider. Needs the same CUDA host stack as `ort-cuda`. |
| `ort-webgpu` | Vendor-neutral GPU (AMD/Intel/NVIDIA) via DX12/Vulkan/Metal. **Experimental upstream**; not shipped as a release asset. |

Windows links no ONNX Runtime at build time and loads it at run time instead
(see the note above), so it has its own pair:

| Feature | Adds |
| --- | --- |
| `ort-dynamic` | ONNX Runtime CPU execution, loading `onnxruntime.dll` at run time. The only way to get ORT at all on Windows. |
| `ort-directml` | Windows GPU on **any DX12 adapter** — Intel, AMD or NVIDIA. Implies `ort-dynamic`; ships the DLL beside the binary. |

The statically linked features above cover every execution provider that has a
prebuilt ONNX Runtime. `ort` also exposes ROCm, OpenVINO and XNNPACK, none of
which has one for a target this repo ships. Loading the runtime dynamically lifts
that restriction — it is how `ort-directml` exists at all — so an OpenVINO or ROCm
DLL is now a plausible route rather than a source build, just an untried one.
Either way each is one line in the execution-provider cascade plus a feature; see
`src/backend/ort.rs`.

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
