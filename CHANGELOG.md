# Changelog

All notable changes to `travsr-embed` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-08-22

Hardening and memory release. Reverses 1.4.0's macOS engine choice on measured
evidence, makes parallel reindex cheap in memory rather than expensive, and
clears the crash residue that could make a killed reindex poison every later run.

### Added
- **`TRAVSR_EMBED_ENGINE` kill-switch.** Setting it to `tract` drops every ORT
  factory from the cascade, forcing the pure-Rust CPU engine regardless of what
  is compiled in or what the catalog says. `auto` (or unset) keeps the normal
  cascade. The switch can only ever remove ORT, so an unrecognised value warns
  and changes nothing.
- **Per-model `macos_engine` in `model.toml`**: `auto` (default) | `tract`
  (per-model kill-switch) | `ort` (opt back into accelerated-first, for a model
  benchmarked faster on CoreML). Defaults to `auto`, so existing `model.toml`
  files need no migration. Setting `tract` for a family tract cannot run
  (ModernBERT, nomic-bert) is a loud error rather than a silent fallback.

  Note: `travsr embed init` rewrites `model.toml` from a closed field set that
  does not yet include `macos_engine`, and does so even when the model files
  were already present, so a hand-set value reverts to `auto` on the next
  `embed init`. Use the env kill-switch for an override that has to survive a
  reinstall. (Tracked for a CLI-side fix; the env switch is unaffected.)
- **`TRAVSR_EMBED_TOKEN_BUDGET`** overrides the derived per-worker padded-token
  budget for tuning. The value is clamped up to one full sequence so a batch can
  always hold a single item.

### Changed
- **macOS now prefers tract over CoreML by default** for families tract can run
  (issue #19). Measured on Apple Silicon with the shipped bge-small model over
  an identical 30k-document corpus, release builds: tract ran about 2x the
  throughput at every point and peaked at 743 MB RSS, finishing in 8m35s, where
  CoreML peaked near 4.0 GB and never finished the corpus. CoreML loses because
  ORT fragments the BERT graph into roughly 97 CoreML/CPU partitions, paying a
  copy at every seam, plus a dynamic-shape recompile tax. The flip is gated to
  macOS, so Linux/CUDA keeps accelerated-first ordering, and `resolve()` is
  split into a testable `resolve_with(is_macos)` so the ordering is
  deterministic on any CI host.
- **Inference tokens are budgeted across all workers, not per worker.** The
  padded-token budget bounds the largest activation tensor **one** worker
  allocates, and every worker runs one batch at a time, so peak sidecar memory
  scaled with `workers x budget`. A fixed per-worker 4,096 therefore made memory
  grow linearly with `-j`, which is what made parallelism expensive: on a
  1,453-node repo, 8 workers peaked at 634 MB against 345 MB for the 2-worker
  default, for only a 1.7x speedup. A total budget of 8,192 is now divided by
  the requested workers, floored at 1,024 and capped at the historical 4,096.
  Memory stays roughly flat as workers scale and throughput improves rather than
  degrades, because large batches mostly buy `BatchLongest` padding waste:
  8 workers measured 45.4s / 316 MB against 48.1s / 634 MB at the old fixed
  budget. One and two workers keep exactly 4,096, so the configurations where
  the old default was already right do not regress. Single-submitter backends
  (GPU/ORT) run one inference loop whatever `-j` says, so they keep the whole
  budget. Batch composition never changes the vectors (attention masking makes
  padding inert, verified bit-identical across budgets on texts spanning 12 to
  480 characters), so this is purely a compute/memory trade-off, not a quality
  one.
- **The serving path memory-maps the HNSW index** (usearch `view`) on Linux and
  macOS instead of copying it into RAM, so the OS pages it in on demand and can
  evict under pressure. A viewed index is immutable, so the lazy-embed add
  becomes a no-op (the vector still persists to `embed.db` and enters the index
  on the next reindex) and reloads re-view. Windows keeps the load fallback,
  where a live file mapping takes a sharing lock that would break the reindex
  sidecar's save.
- **Both reindex paths stop materialising the whole pending corpus.** A `COUNT`
  query provides totals, then 50k rows are fetched, embedded and committed per
  pass. Committed chunks drop out of the `NOT EXISTS` filter, so the loop needs
  no `OFFSET` and always terminates.

### Fixed
- **A KNN index reload transiently doubled index memory** (travsr#736 RCA). The
  reload built the replacement index while the previous one was still alive, a
  transient 2x of the full index size at exactly the moment a just-finished
  reindex had already elevated memory. The old graph is now freed before the
  updated file is loaded, for RAM-copy handles where that 2x is the real
  concern.
- **ONNX Runtime sized its thread pools from the host, ignoring cgroup limits**
  (travsr#736 RCA). Left unset, ORT probes the host's physical core count, so a
  container limited to 2 CPUs on a 64-core host got roughly 64 spinning threads
  and permanent throttling. Intra-op threads are now capped at
  `available_parallelism()` (cgroup quota-aware on Linux) and inter-op at 1.
- **Crash residue from a killed reindex could crash or stall the serving
  sidecar (travsr#735 follow-up).** Four hardening changes:

  1. **Truncated index bytes could SIGSEGV the process.** usearch's native
     `view()`/`load()` parse the file header without validating it, so a
     partially-written `.hnsw.usearch` from a killed run could crash the
     sidecar outright rather than returning an error (reproduced on Linux and
     macOS CI). Every published index holds at least one f32-384 vector and is
     written save-to-tmp + rename, so a file below a 512-byte floor is always
     truncation residue; it is now rejected in Rust before any native parsing,
     on the serve path, the load path, and the KNN reload path.
  2. **A failed index reload was retried on every KNN call, forever**, and a
     rapidly republished file was re-mapped per query — both construct a fresh
     native index per query. Reload attempts are now throttled to one per five
     seconds, and a served (mmap) index keeps answering from its previous
     mapping when a reload is rejected instead of degrading to an empty index.
  3. **A corrupt index was never cleaned up on the load-based reindex path.**
     A direct `--reindex` without `--parallel` failed at load and exited 1 on
     every attempt; that path now quarantines the unloadable file as
     `.corrupt` and rebuilds from embed.db, and stale `.usearch.tmp` files are
     swept. (The `--parallel` path the daemon always uses never loads the
     existing index — it rebuilds and renames at end of run — so it already
     recovered on its own.)
  4. **Undecodable pending rows were silently dropped** by `filter_map(r.ok())`
     and then re-selected by `NOT EXISTS` in every later chunk. Decoding is now
     NULL-tolerant so every selected row is embedded and inserted, residual
     failures are counted and logged, and a chunk consisting only of
     undecodable rows aborts loudly instead of ending the run "successfully"
     with pending work the daemon would respawn forever.

### Notes for upgraders
- **macOS Apple Silicon users will see a backend change, the reverse of
  1.4.0's.** bert-family models now resolve back to `tract` instead of
  `ort/CoreML`, so the first reindex after upgrading logs the existing
  backend-provenance warning and recommends `travsr embed reindex --rebuild`.
  This is the exact mirror of the note 1.4.0 shipped for the tract-to-CoreML
  flip, and the reason is the same: GPU and CPU float accumulation differ in the
  last decimal places, and the warning exists to stop an index being half one
  and half the other. Anyone who reindexed under 1.4.0 on Apple Silicon should
  rebuild. To stay on CoreML instead, set `macos_engine = "ort"` in that model's
  `model.toml`.

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

[1.5.0]: https://github.com/Travsr-com/travsr-embed/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/Travsr-com/travsr-embed/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/Travsr-com/travsr-embed/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/Travsr-com/travsr-embed/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/Travsr-com/travsr-embed/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Travsr-com/travsr-embed/releases/tag/v1.0.0
