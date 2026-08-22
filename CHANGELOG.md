# Changelog

All notable changes to `travsr-embed` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **A killed reindex could poison every later run and turn the serving daemon
  into an allocation loop (travsr#735 follow-up).** Three hardening changes:
  (1) a partial/corrupt `.hnsw.usearch` left by an interrupted run made every
  subsequent reindex fail at load, exit 1, and get respawned by the daemon each
  tick, forever; the reindex write path now quarantines an unloadable index as
  `.corrupt` and rebuilds from embed.db, and stale `.usearch.tmp` files are
  swept. (2) The daemon-mode KNN path retried a failed index reload on every
  single call and re-mapped the file per query when its mtime kept moving; both
  are now throttled to one attempt per 5 seconds, and a served (mmap) index
  keeps answering from the previous mapping when a re-view fails instead of
  degrading to an empty index. (3) Pending rows whose text columns failed to
  decode (unexpected NULL or mistyped values) were silently dropped and then
  re-selected by `NOT EXISTS` in every later chunk; decoding is now
  NULL-tolerant so every selected row is embedded and inserted, residual
  failures are counted and logged, and a chunk consisting only of undecodable
  rows aborts the run loudly instead of ending it "successfully" with pending
  work the daemon would respawn forever.

## [1.4.0] - 2026-08-15

Completes the second half of issue #6. 1.3.0 landed the engine architecture; this
turns it into something users actually get GPU acceleration from, and something CI
actually verifies.

### Added
- **GPU acceleration in shipped assets, on every platform.** macOS **Apple
  Silicon** release binaries now include the CoreML execution provider in the
  DEFAULT asset. ORT's
  macOS prebuilt links it statically, so this costs nothing to install. Linux
  x86_64 gains a separate `travsr-embed-x86_64-unknown-linux-gnu-cuda` asset for
  NVIDIA hosts, with ORT's provider libraries bundled beside the binary
  (`$ORIGIN` rpath). Every shipped file has a `.sha256`. The remaining assets
  (linux aarch64, linux x86_64 default, windows default) are unchanged tract-only
  builds.
- **Windows GPU, for every vendor**, via a new
  `travsr-embed-x86_64-pc-windows-msvc-directml.exe` asset and the `ort-dynamic` /
  `ort-directml` features. Windows cannot link ONNX Runtime at all: the static
  CRT required by `esaxx-rs` conflicts with ORT's prebuilt, and CPU-only ORT fails
  identically because the unresolved `__imp_*` symbols come from ONNX Runtime's own
  object code. These features load `onnxruntime.dll` at run time instead, which
  also lifts the restriction to ORT's prebuilt set and is what makes DirectML
  reachable. DirectML drives any DX12 adapter, so one asset covers Intel, AMD and
  NVIDIA. The DLL is Microsoft's, pinned by version and sha256 at release time.

  Measured on Intel Arc integrated graphics (Core Ultra 7 165U): 92% GPU
  utilisation by the sidecar process where CPU-only measured 0.00%, 200 documents
  in 8.9s against 27.0s on tract (3x), with cross-backend cosine parity
  min 0.999999 / mean 1.000000, inside the 0.999 bound, so the speedup costs no
  accuracy.
- `--capabilities`: prints the compiled engines, the model families they can run,
  and whether an accelerator is compiled in, as JSON. Lets the `travsr` CLI refuse
  a model the installed sidecar cannot execute at selection time rather than
  failing part way through a background reindex. Additive: an older CLI is
  unaffected, and an older sidecar still exits non-zero on the unknown flag, which
  is how the probe detects it.
- `ort-webgpu` feature (vendor-neutral GPU via DX12/Vulkan/Metal, the only
  prebuilt path to AMD/Intel GPUs). Compiled and checked in CI; not yet published
  as a release asset, since upstream marks WebGPU experimental.
- A real ModernBERT test fixture, so "tract cannot run this architecture" and "ORT
  runs it with only a `family` tag" are both verified against an actual RoPE +
  local/global-attention graph instead of a re-tagged BERT model.

### Changed
- **Accelerated batching now bounds tensor shapes.** The accelerated path had
  sequence-length buckets but still built batches with the CPU token-budget
  packer, producing hundreds of distinct `[rows, seq]` shapes, exactly what
  CUDA/TensorRT pay a kernel re-optimisation for. Batches are now grouped so each
  stays within one sequence bucket, and row counts are rounded to a power of two
  capped at that bucket's width. Bulk shapes are unchanged (every row count is
  itself a power of two) while a single query stays one row instead of being
  padded to 128. `embed_query` and the daemon's ad-hoc batches share this path,
  so padding every batch up to the bucket charged the most latency-sensitive path
  127 masked rows for one useful one. The CPU path is untouched.
- ORT backends now always run from a single submit loop, not just accelerated
  ones. `Session::run` takes `&mut self`, so N worker threads under `--parallel`
  serialised on the session mutex anyway: N threads' overhead for one thread's
  throughput. `--parallel` is now documented as a hint for the tract CPU path.
- Removed `TargetInfo`: both fields were dead code and no factory consulted it.
  Platform gating already happens at compile time via the cfg-gated registry.

### CI
- **ORT code is now compiled by CI.** `src/backend/ort.rs` was 300+ lines that no
  job built: not fmt, not clippy, not test. The matrix gains `--features ort`
  (Linux) and `--features ort-coreml` (macOS M1 runners, where the CoreML warm-up
  path runs against real hardware), plus an `ort-cuda` link-check job and a
  non-blocking Windows `ort-cuda` probe for the static-CRT question that gates a
  Windows GPU asset.
- The plumbing for the `#[ignore]` model-backed tests (cross-backend parity,
  ModernBERT, CoreML) is in place: the ORT rows stage cached fixtures and run
  them. The tests stay skipped (loudly, naming what went uncovered) until the
  fixtures release exists and its checksums are recorded in
  `.github/scripts/fetch-test-models.sh`.
- `cargo-deny` runs with `--all-features`, so the ORT dependency tree the release
  assets ship is license- and advisory-checked.
- MSRV job also checks `--features ort`, keeping ort's own MSRV claim honest.
- Release builds smoke-test each artifact (`--version` + `--capabilities`) and
  fail if a default asset unexpectedly produces ORT runtime libraries.

### Notes for upgraders
- **macOS Apple Silicon users will see a backend change.** bert-family models now resolve to
  `ort/CoreML` instead of `tract` (higher preference), so the first reindex after
  upgrading logs the existing backend-provenance warning and recommends
  `travsr embed reindex --rebuild`. That is intended: GPU and CPU float
  accumulation differ in the last decimal places, and the warning exists to stop
  an index being half one and half the other.

## [1.3.0] - 2026-08-12

### Added
- `EmbedBackend` trait and a capability-based engine cascade (issue #6): backend
  selection is now a capability match instead of a fixed engine. Adds an
  `OrtFactory` (feature-gated) that walks the compiled execution-provider
  cascade (CUDA to TensorRT to CoreML) with warm-up confirmation, alongside the
  existing `TractFactory`. `embed.db` now records backend provenance and warns
  when the backend changes.
- Windows release target: `x86_64-pc-windows-msvc` added to the release build
  matrix, so `travsr embed init` on Windows resolves instead of 404ing.
- Build-only Windows CI job that runs the release build (and, per review, the
  test suite) on every PR instead of only when a tag runs `release.yml`.

### Fixed
- Windows build: define `MAP_FAILED` for the usearch 2.24.0 build under MSVC,
  and link the static CRT (`+crt-static`) so esaxx-rs's `/MT` objects stop
  conflicting with Rust's default `/MD` runtime.
- Post-merge compile fixes after the doc-space merge: deduped
  `EmbedPlugin::plugin_version`, and ported `embed-jsonl` from the removed
  `EncoderModel` onto `backend::create_backend` so it shares the engine
  cascade and stays byte-identical to the sidecar's vectors.

### CI
- `cargo-deny` now actually runs (checked out against the `travsr` sibling it
  needs for the `[patch.crates-io]` path deps) instead of being decorative
  config.
- Preflight no longer misattributes this repo's own compile errors to a
  missing sibling PR; added `permissions: contents: read`, a concurrency
  group, and cache keys scoped to `Cargo.lock` so a sibling-repo lockfile
  change stops evicting this repo's build cache.

## [1.2.0] - 2026-08-02

### Added
- Doc-space HNSW index and a query-embedding memo, sized by the freshness gap
  it needs to span (content-addressed embedding freshness).
- `deny.toml`, and fixed the two advisories it surfaced.

### Changed
- Bounded invalidation memory and time-based flush for the freshness tracker,
  with test coverage for kill/recovery.

## [1.1.0] - 2026-07-06

### Added
- Graceful reindex cancel via a cancel sentinel (WS3, #420).

## [1.0.0] - 2026-07-05

Initial release of the `travsr-embed` sidecar (RFC-018).

### Added
- Embedding sidecar binary with a SQLite-backed (`embed.db`) node/symbol
  embedding store and a usearch HNSW vector index.
- Multi-model support (`bge-small`/`base`/`large`) selected at runtime via
  `--model-id`; package and binary renamed from `travsr-embed-nomic` to
  `travsr-embed`.
- `tract` ONNX backend with multi-threaded inference (RFC-021).
- Parallel reindex worker with a shared atomic batch queue (Kafka-style
  consumer group) and row-count partitioning to fix load imbalance.
- CDC tombstone apply and `node_embeddings` persistence (RFC-019).
- Symbol-level KNN with an exclusion list and a two-phase build split
  (RFC-018 step 10).
- Lazy on-demand embedding via BM25 fallback, including an OR-semantics fix
  for multi-word FTS queries.
- CI, gcc-11 build fix, and macOS/Linux release packaging so the sidecar
  binary can actually ship.

### Changed
- Relicensed from MIT to Apache-2.0.

[1.3.0]: https://github.com/Travsr-com/travsr-embed/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/Travsr-com/travsr-embed/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/Travsr-com/travsr-embed/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Travsr-com/travsr-embed/releases/tag/v1.0.0
