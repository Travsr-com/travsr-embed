# Changelog

All notable changes to `travsr-embed` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
