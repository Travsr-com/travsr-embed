# travsr-embed

Embedding sidecar for [Travsr](https://github.com/Travsr-com/travsr). Encodes graph
nodes and symbols into vectors with the BGE model family (and, with the `ort`
feature, other ONNX model families) and maintains a disk-persistent usearch HNSW
index for nearest-neighbour retrieval.

The `travsr` CLI downloads a prebuilt `travsr-embed` binary per platform from this
repo's [GitHub Releases](https://github.com/Travsr-com/travsr-embed/releases); most
users never build it directly.

## Which release asset do I want?

The default asset for your platform always works, and GPU assets are opt-in.
Intel Macs are CPU-only: ONNX Runtime publishes no prebuilt for
`x86_64-apple-darwin`, so no ORT-based engine can be built for them at all.

Apple Silicon is the exception in both directions. CoreML is compiled into the
default asset (statically linked, nothing to install), but as of 1.5.0 it is no
longer what runs by default: for model families `tract` can run, tract is both
faster and far cheaper in memory than CoreML on this hardware, so it is
preferred. See [Choosing an engine on macOS](#choosing-an-engine-on-macos).

| Your machine | Asset | Accelerator | Host prerequisite |
| --- | --- | --- | --- |
| macOS (Apple Silicon) | default | none by default (tract, CPU); CoreML for families tract cannot run, or opt in per model | none |
| macOS (Intel) | default | none (tract, CPU) | none |
| Linux x86_64 + NVIDIA GPU | `…-x86_64-unknown-linux-gnu-cuda` | CUDA | CUDA runtime + cuDNN, glibc 2.38+, and a Haswell-or-newer CPU (`x86-64-v3`) |
| Windows x86_64 + any GPU | `…-x86_64-pc-windows-msvc-directml.exe` | DirectML (Intel, AMD **or** NVIDIA) | a DirectX 12 GPU |
| Linux x86_64, no GPU | default | none (tract, CPU) | none |
| Linux aarch64 (incl. OCI) | default | none (tract, CPU) | none |
| Windows x86_64, no GPU | default | none (tract, CPU) | none |

Accelerated assets ship their ONNX Runtime libraries alongside the binary
(`libonnxruntime_providers_*.so` for `-cuda`, `onnxruntime.dll` for `-directml`).
**Keep them in the same directory as the binary.** `-cuda` finds them via an
`$ORIGIN` rpath, and `-directml` loads its DLL at run time from beside the
executable (or from `ORT_DYLIB_PATH`). Each shipped file has a `.sha256` sidecar.

The glibc 2.38 floor on `-cuda` is inherited from ORT's prebuilt, which
references `__isoc23_*` symbols added in that release. It rules out Ubuntu 22.04,
Debian 12 and RHEL 9. Use the default CPU asset there, which has no such floor.

Nothing silently degrades: if a GPU asset cannot initialize its accelerator (no
driver, wrong CUDA version, missing provider libraries), the sidecar logs the
reason and falls back to CPU inference rather than failing. The worst case of
downloading the wrong asset is CPU speed, not a broken install.

Windows deserves a note, because it is built differently from every other GPU
asset. No ONNX Runtime is linked into it at all: the static CRT this repo needs
for `esaxx-rs` conflicts with ORT's prebuilt, and that blocks *every* statically
linked ORT feature there, CPU-only included, since the unresolved `__imp_*`
symbols come from ONNX Runtime's own object code. The `-directml` asset sidesteps
that by loading `onnxruntime.dll` at run time instead, which is also what frees it
from ORT's prebuilt set and makes DirectML reachable in the first place.

Not yet available, and why:

- **Intel NPU** (and any other OpenVINO-only target): OpenVINO is the sole route
  to an NPU, and it has no prebuilt ONNX Runtime for any target here. Reaching it
  means building ONNX Runtime from source. Now that Windows loads its runtime
  dynamically the constraint is softer than it looks (an OpenVINO-enabled DLL
  would be the same shape of solution as the DirectML one), but nobody has tried
  it.
- **ROCm**: same story, a Cargo feature exists upstream, no prebuilt does.
- **WebGPU**: the portable path (D3D12 / Vulkan / Metal, so any modern adapter).
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
| `ort-directml` | Windows GPU on **any DX12 adapter**: Intel, AMD or NVIDIA. Implies `ort-dynamic`; ships the DLL beside the binary. |

The statically linked features above cover every execution provider that has a
prebuilt ONNX Runtime. `ort` also exposes ROCm, OpenVINO and XNNPACK, none of
which has one for a target this repo ships. Loading the runtime dynamically lifts
that restriction (it is how `ort-directml` exists at all), so an OpenVINO or ROCm
DLL is now a plausible route rather than a source build, just an untried one.
Either way each is one line in the execution-provider cascade plus a feature; see
`src/backend/ort.rs`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for which of these can be built on which
host (notably: no ORT feature builds on a Windows *GNU* toolchain).

Backend selection at runtime is capability-based: each registered backend
declares which model families it can run, and the sidecar picks the first
backend, in preference order, that can serve the model actually being loaded.
Accelerated engines come first, except on macOS (see below).

### Choosing an engine on macOS

Since 1.5.0, macOS prefers `tract` over CoreML for families tract can run.
Measured on Apple Silicon with the shipped bge-small model over an identical
30k-document corpus, release builds: tract ran about 2x the throughput at every
point and peaked at 743 MB RSS, finishing in 8m35s, where CoreML peaked near
4.0 GB and never finished the corpus. CoreML loses because ORT fragments the
BERT graph into roughly 97 CoreML/CPU partitions, paying a copy at every seam,
plus a dynamic-shape recompile tax. The flip is gated to macOS, so Linux/CUDA
and Windows/DirectML keep accelerated-first ordering.

CoreML is still reached on macOS for families `tract` cannot run (ModernBERT,
nomic-bert), and can be opted back into per model with `macos_engine` in that
model's `model.toml`:

| `macos_engine` | Effect on macOS |
| --- | --- |
| `auto` (default) | Prefer `tract` for families it can run; fall through to ORT/CoreML otherwise. |
| `tract` | Force `tract`, dropping every ORT engine. A loud error for a family tract cannot run, rather than a silent fallback. |
| `ort` | Restore accelerated-first ordering. Set this only for a model benchmarked faster on CoreML. |

It defaults to `auto`, so existing `model.toml` files need no migration, and it
has no effect off macOS.

**`macos_engine` does not survive a reinstall.** `travsr embed init` rewrites
`model.toml` from a closed field set that does not yet include it, and does so
even when the model files were already present, so a hand-set value reverts to
`auto` on the next `embed init`. For an override that has to persist, use the
`TRAVSR_EMBED_ENGINE` environment variable instead.

### Environment variables

| Variable | Effect |
| --- | --- |
| `TRAVSR_EMBED_ENGINE` | `tract` drops every ORT engine (both accelerated and ORT CPU) from the cascade, forcing the pure-Rust CPU engine regardless of what is compiled in or what the catalog says. `auto`, or unset, keeps the normal cascade. The switch can only ever remove ORT, so an unrecognised value warns and changes nothing. Also reflected in `--capabilities`, so the handshake never advertises acceleration that has been switched off. |
| `TRAVSR_EMBED_TOKEN_BUDGET` | Overrides the derived per-worker padded-token budget, for tuning. By default a total budget of 8,192 is divided across the requested `--parallel` workers, floored at 1,024 and capped at 4,096, which keeps peak memory roughly flat as workers scale. Single-submitter backends (GPU/ORT) keep the whole budget. |
| `ORT_DYLIB_PATH` | Where the `-directml` (and any `ort-dynamic`) build loads `onnxruntime.dll` from, when not beside the executable. |

### Reporting what a build can do

```bash
travsr-embed --capabilities
```

Prints the compiled engines, the model families they can run, and whether an
accelerator is compiled in, as JSON. The `travsr` CLI uses this to refuse a model
the installed sidecar cannot execute at *selection* time, instead of failing part
way through a background reindex. `accelerated_compiled` reports what was
compiled, not whether this machine's GPU will actually be confirmed at load time.
That answer needs a real model load, and appears in the startup log.

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
