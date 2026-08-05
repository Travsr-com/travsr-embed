// travsr-embed-nomic — RFC-018 embedding sidecar.
//
// Modes controlled by argv:
//
//   (no args / --db-path)      Daemon mode: speak EmbedPlugin IPC over stdio.
//                              Requires --db-path <graph.db> so the sidecar
//                              knows which repo's per-repo HNSW index to load.
//
//   --reindex <db>             One-shot: embed all pending nodes in graph.db,
//                              write node_embeddings rows to embed.db, and
//                              update the per-repo HNSW index. Exits when done.
//
//   --reindex <db> --embed-db <embed.db>
//                              Same as above with an explicit embed.db path.
//                              Defaults to <db's dir>/embed.db when omitted.
//
//   --reindex <db> --shard <i>/<n>
//                              One-shot shard: embed only nodes where id % n = i.
//                              Skips HNSW writes — the CLI orchestrator calls
//                              --rebuild-index when all shards complete.
//
//   --rebuild-index <db>       Rebuild per-repo HNSW from embed.db.node_embeddings.
//                              No ONNX inference — pure SQLite stream.
//
// RFC-019: node_embeddings lives in embed.db (sibling of graph.db), not graph.db.
// graph.db is opened read-only for node queries; embed.db is opened with
// synchronous=OFF + wal_autocheckpoint=0 for fast bulk writes (~8-15x faster).
//
// HNSW index placement: <db-path's dir>/<MODEL_ID>.hnsw.usearch
// (co-located with graph.db so every repo has its own index; node IDs are
//  SQLite rowids that are only unique within one db).

#![forbid(unsafe_code)]

mod backend;
mod encode;
mod freshness;
mod index;
mod model;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use anyhow::{Context as _, Result};
use rusqlite::Connection;
use travsr_plugin_protocol::{
    EmbedPlugin, EmbedRequest, EmbedResponse, KnnRequest, KnnResponse, Space,
};
use travsr_plugin_sdk::run_embed_plugin;

/// Human-readable backend label shown in `travsr embed status` and sidecar logs.
/// `dim` comes from the model descriptor — never hardcoded per model id.
fn backend_label(model_id: &str, dim: usize) -> String {
    format!("{model_id} fp32 dim-{dim}")
}
// MAX_BATCH: hard cap on items per forward pass.
// TOKEN_BUDGET: soft cap on padded tensor cost (BatchLongest pads all items to
// max_seq_in_batch; cost = max_seq × count). At TOKEN_BUDGET=4096 the hidden
// state tensor is 4096×384×4B ≈ 6 MB regardless of batch count.
// For our workload (avg ~12 tokens/node), 4096 tokens ≈ 341 nodes — well under
// MAX_BATCH=512, so the budget is the effective limit in practice.
const MAX_BATCH: usize = 512;
const TOKEN_BUDGET: usize = 4_096;
// Commit to embed.db every TX_BATCH rows. Kept small so embed.db reflects
// progress in near-real-time (the CLI progress bar and `embed status` poll it).
// Cheap: embed.db uses synchronous=OFF, so a commit is a WAL append, not an
// fsync — the single fsync happens once at the end via wal_checkpoint(TRUNCATE).
const TX_BATCH: usize = 500;
/// Commit a partially-filled buffer once it is this old, independent of how many
/// rows it holds (#376 O2 / G3).
///
/// `TX_BATCH` alone does not deliver the "near-real-time progress" the comment
/// above promises: a pass with fewer than `TX_BATCH` items commits **nothing**
/// until it finishes. The host's no-progress watchdog
/// (`travsr_plugin_host::embed_catalog`, `NO_PROGRESS_SECS = 600`) watches
/// exactly this row count, so such a pass looks stalled for its entire duration
/// and is killed at 600 s with zero rows written — an error the user cannot
/// action, on a pass that was making perfectly good progress.
///
/// Reachable rather than theoretical: doc chunks embed at roughly 2/s (plan
/// §8.7), so 499 of them take ~250 s on an idle machine, but under
/// `embed.priority = low`/`idle` or on loaded hardware the same pass crosses
/// 600 s. #376 W2 made sub-`TX_BATCH` invalidation passes the *common* case.
///
/// 30 s leaves a 20x margin under the watchdog while costing one extra WAL
/// append per 30 s of work, which is not measurable next to inference.
const TX_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// Whether a buffer holding `buffered` rows, last committed at `last_flush`,
/// should be committed now (#376 O2).
///
/// Split out from the two write loops it governs so the property that actually
/// matters can be asserted directly rather than inferred from a long-running
/// integration run: **a non-empty buffer always becomes observable within
/// `TX_FLUSH_INTERVAL`, whatever its size.** An empty buffer never flushes —
/// committing nothing would reset the timer without moving the row count the
/// watchdog reads, which is the bug wearing a different hat.
fn should_flush(buffered: usize, last_flush: std::time::Instant) -> bool {
    if buffered == 0 {
        return false;
    }
    buffered >= TX_BATCH || last_flush.elapsed() >= TX_FLUSH_INTERVAL
}

/// Which nodes to embed in a reindex run.
#[derive(Clone, Copy)]
enum Phase {
    /// All pending nodes (default `--reindex` with no phase flag).
    All,
    /// Only nodes with `shell_number >= threshold` — high-centrality fast pass.
    Phase1(u32),
    /// Only nodes with `shell_number < threshold` — background sweep.
    /// Skips inline HNSW updates and rebuilds the full index at the end so it
    /// includes both Phase 1 and Phase 2 nodes without a HNSW file race.
    Phase2(u32),
}

// ── Plugin struct ─────────────────────────────────────────────────────────────

/// #376 Phase 2: single-slot memo of the last query's embedding. The host
/// issues one round trip per space (Space::Code, then Space::Docs) for the same
/// `query_text` (§4.4's "one inference, two searches", achieved here instead of
/// by fusing both spaces into one wire request — see `Space`'s doc comment in
/// travsr-plugin-protocol). The memo is what makes that contract true, so its
/// lifetime has to exceed the gap between the two round trips.
struct QueryEmbedCache {
    text: String,
    blob: Vec<u8>,
    at: std::time::Instant,
}

impl QueryEmbedCache {
    /// Whether this entry may serve `query_text` at `age`.
    ///
    /// Split out from [`NomicPlugin::embed_query_cached`] so the TTL's effect is
    /// testable without a loaded model. It is worth testing directly: the
    /// k8s Gate 4 failure it fixes only reproduces when the machine is loaded
    /// enough to put >5 s between the two lanes, so a green gate run is
    /// consistent with the fix but does not prove it.
    fn serves(&self, query_text: &str, age: std::time::Duration) -> bool {
        self.text == query_text && age < QUERY_EMBED_CACHE_TTL
    }
}

/// How long a memoized query embedding stays usable.
///
/// **Sized by the gap it must span, not by staleness risk.** The two round trips
/// are not "milliseconds apart" as this was originally written for: the code
/// lane runs a cross-encoder rerank between them, which is ~1 s on a small repo
/// and multiple seconds on a large one. Measured on kubernetes (264 k nodes),
/// the slowest queries put 6.1-7.1 s between the two calls, so the previous 5 s
/// bound expired mid-query and the docs lane re-embedded the same text: 5 of 20
/// queries did two inferences, failing the §7 Gate 4 contract on that repo while
/// passing on travsr, where the same queries cost about a second. Scale, not
/// correctness, is what moved.
///
/// There is no staleness for a longer bound to risk. The entry is keyed on exact
/// query text, an embedding is a pure function of (text, model), and a sidecar
/// process loads exactly one model for its lifetime — so a hit is always the
/// vector this process would have recomputed. The TTL is kept only as a bound on
/// how long one 3 KB blob may sit in memory, and is set an order of magnitude
/// above the worst interval measured.
const QUERY_EMBED_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(60);

/// #376 Phase 2: set to a **file path** to record per-KNN memo hit/miss events.
///
/// The memo is what makes the plan's §4.4 "one inference, two searches"
/// guarantee true after the wire protocol dropped `KnnRequest.spaces` fusion in
/// favour of one round trip per space. Wall-clock latency cannot verify that
/// guarantee: it is dominated by the cross-encoder passes downstream, so a
/// silently-missing memo (a full 200-270ms re-embed against a 600ms circuit
/// breaker) is invisible in a timing measurement. That is not hypothetical —
/// the host sent the raw query on the docs lane and the normalized one on the
/// code lane, so every punctuated query missed, undetected, until inference
/// count was made observable. This makes it observable.
///
/// A file rather than stderr because the plugin host captures sidecar stderr
/// into a bounded ring buffer for error surfacing (`StderrRing`,
/// `travsr-plugin-host/src/embed_sidecar.rs`) — it never reaches the host
/// process's own stderr, so a bench harness cannot read it there.
///
/// Off unless the variable is set, and then costs one `OnceLock` read per KNN.
/// Consumed by `bench/run-phase2-gate.mjs`'s single-inference gate.
const QUERY_CACHE_DEBUG_ENV: &str = "TRAVSR_EMBED_QUERY_CACHE_DEBUG";

/// Resolved once — the sidecar is long-lived and the env cannot change under it.
/// An empty value counts as unset so `FOO=` behaves like absence.
fn query_cache_debug_path() -> Option<&'static Path> {
    static PATH: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var(QUERY_CACHE_DEBUG_ENV)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from)
    })
    .as_deref()
}

struct NomicPlugin {
    model: Arc<dyn backend::EmbedBackend>,
    model_id: String,
    backend: String,
    /// Code-space HNSW index — None until first KNN call if not present at startup.
    index: Mutex<Option<index::VecIndex>>,
    index_path: PathBuf,
    /// #376 Phase 2: doc-space HNSW index, mirrors `index`/`index_path`. None
    /// both before the first reindex and on every repo with no markdown.
    doc_index: Mutex<Option<index::VecIndex>>,
    doc_index_path: PathBuf,
    /// graph.db path — for FTS candidate lookup in lazy embed path.
    db_path: PathBuf,
    /// embed.db path — for NOT-EXISTS filter + async persist of lazy embeds.
    embed_db_path: PathBuf,
    query_embed_cache: Mutex<Option<QueryEmbedCache>>,
}

impl NomicPlugin {
    /// `model_dir`  — global model directory (ONNX + tokenizer files)
    /// `index_path` — per-repo HNSW file (co-located with graph.db)
    /// `db_path`    — graph.db (for FTS candidate lookup in lazy embed path)
    /// `model_id`   — catalog ID, e.g. "bge-small-en-v1.5"
    fn load(
        model_dir: &Path,
        index_path: PathBuf,
        db_path: PathBuf,
        model_id: &str,
    ) -> Result<Self> {
        let embed_db_path = embed_db_path_for(&db_path);
        let desc = model::ModelDescriptor::load(model_dir).context("loading model descriptor")?;
        let dim = desc.output_dim();
        let model = backend::create_backend(model_dir, &desc).context("loading model")?;
        // Issue #6: GPU fp32 and tract fp32 are not bit-identical. If embed.db
        // was produced by a different engine, lazy embeds from this one would be
        // mixed-provenance — warn and recommend a rebuild.
        if let Some(prev) = read_backend_provenance(&embed_db_path) {
            if prev != model.backend_name() {
                tracing::warn!(
                    previous = %prev,
                    current = model.backend_name(),
                    "embed.db was built by a different backend — run \
                     `travsr embed reindex --rebuild` for consistent embeddings"
                );
            }
        }
        let index = index::VecIndex::try_load(&index_path).unwrap_or_else(|e| {
            tracing::warn!(
                "could not load HNSW index: {e:#} — KNN disabled until `travsr embed reindex` runs"
            );
            None
        });
        let doc_index_path = doc_index_path_for_db(&db_path, model_id);
        let doc_index = index::VecIndex::try_load(&doc_index_path).unwrap_or_else(|e| {
            tracing::debug!("no doc-space HNSW index yet: {e:#}");
            None
        });
        Ok(Self {
            model,
            model_id: model_id.to_owned(),
            backend: backend_label(model_id, dim),
            index: Mutex::new(index),
            index_path,
            doc_index: Mutex::new(doc_index),
            doc_index_path,
            db_path,
            embed_db_path,
            query_embed_cache: Mutex::new(None),
        })
    }

    /// Embed `query_text`, reusing the single-slot memo when the immediately
    /// preceding call embedded the same text within `QUERY_EMBED_CACHE_TTL`.
    ///
    /// `space` is used only for the [`QUERY_CACHE_DEBUG_ENV`] trace line; the
    /// memo itself is space-agnostic on purpose, since sharing one query
    /// embedding across both spaces is the entire point (§4.4).
    fn embed_query_cached(&self, query_text: &str, space: Space) -> Result<Vec<u8>> {
        if let Ok(guard) = self.query_embed_cache.lock() {
            if let Some(entry) = guard.as_ref() {
                if entry.serves(query_text, entry.at.elapsed()) {
                    self.trace_query_cache(space, "hit", query_text);
                    return Ok(entry.blob.clone());
                }
            }
        }
        self.trace_query_cache(space, "miss", query_text);
        let blob = self.model.embed_query(query_text)?;
        if let Ok(mut guard) = self.query_embed_cache.lock() {
            *guard = Some(QueryEmbedCache {
                text: query_text.to_string(),
                blob: blob.clone(),
                at: std::time::Instant::now(),
            });
        }
        Ok(blob)
    }

    /// One tab-separated line per KNN when [`QUERY_CACHE_DEBUG_ENV`] is set.
    /// A `miss` is one real query-embedding inference; the §4.4 contract is
    /// exactly one `miss` per distinct query regardless of how many spaces are
    /// searched. Tabs (not spaces) so a query containing spaces stays one
    /// field, and the query goes last so a query containing a tab cannot shift
    /// the fields the gate parses.
    ///
    /// Diagnostics only: every failure is swallowed, since a bench trace must
    /// never be able to fail a real KNN. Opened in append mode per call rather
    /// than held open — this path runs at most twice per query and a held
    /// handle would outlive the harness's own truncation of the file.
    fn trace_query_cache(&self, space: Space, outcome: &str, query_text: &str) {
        let Some(path) = query_cache_debug_path() else {
            return;
        };
        let space = match space {
            Space::Code => "code",
            Space::Docs => "docs",
        };
        // Single newline-terminated write; O_APPEND keeps concurrent sidecars
        // from interleaving partial lines.
        let line = format!("QUERY_EMBED_CACHE\t{space}\t{outcome}\t{query_text}\n");
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            use std::io::Write as _;
            let _ = f.write_all(line.as_bytes());
        }
    }
}

impl EmbedPlugin for NomicPlugin {
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn embedding_dim(&self) -> u32 {
        self.model.dim() as u32
    }
    fn backend(&self) -> &str {
        &self.backend
    }
    // #376 Phase 2: must be evaluated here, in travsr-embed's own source, not
    // in the shared trait default or the SDK runner — env! expands against the
    // lexically containing crate's Cargo.toml, so only this crate's own
    // CARGO_PKG_VERSION reports travsr-embed's real release version (see
    // EmbedPlugin::plugin_version's doc comment for the bug this avoids).
    fn plugin_version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }
    fn max_batch(&self) -> u32 {
        MAX_BATCH as u32
    }

    fn embed_batch(&self, req: &EmbedRequest) -> EmbedResponse {
        let texts: Vec<&str> = req.texts.iter().map(String::as_str).collect();
        match self.model.embed_documents(&texts) {
            Ok(blobs) => EmbedResponse { embeddings: blobs },
            Err(e) => {
                tracing::error!("embed_batch failed: {e:#}");
                EmbedResponse { embeddings: vec![] }
            }
        }
    }

    fn knn(&self, req: &KnnRequest) -> KnnResponse {
        match self.knn_impl(req) {
            Ok((ids, scores)) => KnnResponse {
                node_ids: ids,
                scores,
            },
            Err(e) => {
                tracing::warn!("knn failed (non-fatal): {e:#}");
                KnnResponse {
                    node_ids: vec![],
                    scores: vec![],
                }
            }
        }
    }
}

impl NomicPlugin {
    /// #376 Phase 2: dispatch on which index this request searches.
    fn knn_impl(&self, req: &KnnRequest) -> Result<(Vec<i64>, Vec<f32>)> {
        match req.space {
            Space::Code => self.knn_impl_code(req),
            Space::Docs => self.knn_impl_docs(req),
        }
    }

    /// #376 Phase 2: doc-space KNN. No lazy-embed fallback (plan §4.4: "Never
    /// lazy-embed the doc space") — a doc corpus is 2-4 orders of magnitude
    /// smaller than the code corpus (§8.7), so `knn_raw.len() < req.k` is the
    /// routine case here, not the rare one the code path's gate protects
    /// against; running the ~15s FTS wedge (`fts_candidates_unembedded`) on
    /// nearly every doc query would reintroduce the exact daemon stall #391
    /// fixed on the code path.
    fn knn_impl_docs(&self, req: &KnnRequest) -> Result<(Vec<i64>, Vec<f32>)> {
        let query_blob = self.embed_query_cached(&req.query_text, Space::Docs)?;
        let mut guard = self
            .doc_index
            .lock()
            .map_err(|_| anyhow::anyhow!("doc index mutex poisoned"))?;

        if guard.is_none() && self.doc_index_path.exists() {
            *guard = index::VecIndex::try_load(&self.doc_index_path)?;
        }

        let raw: Vec<(i64, f32)> = match guard.as_mut() {
            None => {
                tracing::debug!("no doc-space HNSW index — run `travsr embed reindex`");
                vec![]
            }
            Some(idx) => idx.knn(&query_blob, req.k)?,
        };

        let ids = raw.iter().map(|(id, _)| *id).collect();
        let scores = raw
            .iter()
            .map(|(_, dist)| (1.0 - dist).clamp(0.0, 1.0))
            .collect();
        Ok((ids, scores))
    }

    fn knn_impl_code(&self, req: &KnnRequest) -> Result<(Vec<i64>, Vec<f32>)> {
        let query_blob = self.embed_query_cached(&req.query_text, Space::Code)?;
        let query_vec = model::blob_to_f32(&query_blob);

        // ── KNN against HNSW (Phase 1 nodes) ─────────────────────────────
        // Hold the mutex only for the index operation; release before the
        // lazy embed path so we don't block other KNN calls during inference.
        let knn_raw: Vec<(i64, f32)> = {
            let mut guard = self
                .index
                .lock()
                .map_err(|_| anyhow::anyhow!("index mutex poisoned"))?;

            // Late-load: daemon may start before the first reindex run.
            if guard.is_none() && self.index_path.exists() {
                *guard = index::VecIndex::try_load(&self.index_path)?;
            }

            match guard.as_mut() {
                None => {
                    tracing::debug!("no HNSW index — run `travsr embed reindex`");
                    vec![]
                }
                Some(idx) => idx.knn(&query_blob, req.k)?,
            }
        };

        // ── Lazy embed: BM25 fallback for un-embedded nodes ───────────────
        // Find nodes that matched the FTS index but haven't been embedded yet,
        // embed them on-the-fly (~20-50ms for 10-20 nodes), add to the
        // in-memory HNSW, and persist to embed.db asynchronously.
        //
        // PERF (measured on kubernetes, 261k nodes, 131k embeddings @ 100%):
        // the FTS candidate query (`fts_candidates_unembedded`) costs ~15 s on a
        // large, fully-embedded repo and returns *nothing* — every node already
        // has an embedding, so the only result of running it is latency. That
        // synchronous cost on the hot KNN path was the root cause of:
        //   • daemon queries taking ~15 s (the 600 ms host hook timeout returns
        //     empty while this query still runs, holding the sidecar lock → the
        //     "wedge"),
        //   • embed_used == false (the host circuit-breaker discards any KNN that
        //     overruns its budget),
        //   • slow cold-path `travsr ask`.
        //
        // Gate: lazy embed only adds value when HNSW under-delivers (a sparse or
        // partially-built index). When HNSW already returned the full `k` set we
        // skip it entirely — the seeds are already the strongest available.
        let lazy_scored = if knn_raw.len() >= req.k as usize {
            vec![]
        } else {
            self.lazy_embed_candidates(&req.query_text, &query_vec)
                .unwrap_or_else(|e| {
                    tracing::debug!("lazy embed skipped (non-fatal): {e:#}");
                    vec![]
                })
        };

        // ── Merge: KNN first (higher confidence), then lazy, dedup ───────
        let mut seen: HashSet<i64> = HashSet::new();
        let mut ids: Vec<i64> = Vec::with_capacity(req.k as usize);
        let mut scores: Vec<f32> = Vec::with_capacity(req.k as usize);

        for (id, dist) in knn_raw {
            if seen.insert(id) {
                ids.push(id);
                // usearch returns cosine distance → convert to similarity.
                scores.push((1.0 - dist).clamp(0.0, 1.0));
            }
        }
        for (id, sim) in lazy_scored {
            if seen.insert(id) && ids.len() < req.k as usize {
                ids.push(id);
                scores.push(sim);
            }
        }

        Ok((ids, scores))
    }

    /// BM25/FTS fallback: find un-embedded candidates matching the query,
    /// embed them on-the-fly, add to in-memory HNSW, persist to embed.db.
    ///
    /// Returns (node_id, cosine_similarity) pairs for the newly embedded nodes.
    /// All errors are treated as non-fatal — the caller falls back to KNN-only.
    fn lazy_embed_candidates(
        &self,
        query_text: &str,
        query_vec: &[f32],
    ) -> Result<Vec<(i64, f32)>> {
        // Skip if embed.db doesn't exist (first-run before any reindex)
        if !self.embed_db_path.exists() {
            return Ok(vec![]);
        }

        let candidates = self.fts_candidates_unembedded(query_text, 20)?;
        if candidates.is_empty() {
            return Ok(vec![]);
        }

        tracing::debug!(n = candidates.len(), "lazy embed: on-the-fly embedding");

        let texts: Vec<&str> = candidates.iter().map(|(_, t)| t.as_str()).collect();
        let blobs = self.model.embed_documents(&texts)?;

        // Add to in-memory HNSW + compute similarity against query
        let mut results: Vec<(i64, f32)> = Vec::with_capacity(candidates.len());
        {
            let guard = self
                .index
                .lock()
                .map_err(|_| anyhow::anyhow!("index mutex poisoned"))?;
            if let Some(ref idx) = *guard {
                for ((nid, _), blob) in candidates.iter().zip(blobs.iter()) {
                    let vec = model::blob_to_f32(blob);
                    // BGE CLS vectors are unit-normalised → dot product = cosine similarity
                    let sim: f32 = vec
                        .iter()
                        .zip(query_vec.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        .clamp(0.0, 1.0);
                    results.push((*nid, sim));
                    let _ = idx.add(*nid, &vec); // skip-if-present is safe
                }
            }
        }

        // Persist to embed.db in a background thread so the hot query path
        // is not blocked by SQLite I/O. INSERT OR IGNORE is safe under races.
        let edb = self.embed_db_path.clone();
        let mid = self.model_id.clone();
        // #376 W3: carry the content hash of the exact text embedded, so a
        // lazily-embedded vector is verifiable like any other.
        let pairs: Vec<(i64, Vec<u8>, String)> = candidates
            .iter()
            .zip(blobs)
            .map(|((nid, text), blob)| (*nid, blob, crate::freshness::text_hash(text)))
            .collect();
        std::thread::Builder::new()
            .name("lazy-embed-persist".into())
            .spawn(move || {
                if let Err(e) = persist_lazy_embeddings(&edb, &pairs, &mid) {
                    tracing::warn!("lazy embed persist failed (non-fatal): {e:#}");
                }
            })
            .ok();

        Ok(results)
    }

    /// Query the FTS5 trigram index for nodes relevant to `query_text` that
    /// do not yet have an embedding in embed.db. Returns up to `limit` pairs
    /// of (node_id, "kind: signature") ready for on-the-fly embedding.
    fn fts_candidates_unembedded(
        &self,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<(i64, String)>> {
        // Extract words ≥4 chars for a focused FTS5 trigram MATCH.
        // Wrap each in double-quotes for FTS5 phrase semantics (exact substring).
        // Take the 3 longest words to keep the query specific but not too narrow.
        let mut words: Vec<&str> = query_text
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.len() >= 4)
            .collect();
        words.sort_unstable_by_key(|w| std::cmp::Reverse(w.len()));
        words.dedup();
        // OR between terms: any matching term is a lazy-embed candidate.
        // AND would require every query word to appear in one signature — too
        // strict for multi-word queries where words like "user" or "input" never
        // co-occur with "validate" in the same function name.
        let fts_query: String = words
            .iter()
            .take(3)
            .map(|w| format!("\"{w}\""))
            .collect::<Vec<_>>()
            .join(" OR ");

        if fts_query.is_empty() {
            return Ok(vec![]);
        }

        let conn = Connection::open_with_flags(
            &self.db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("lazy embed: open graph.db")?;

        let embed_str = self
            .embed_db_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("embed.db path not UTF-8"))?;
        let escaped = embed_str.replace('\'', "''");
        conn.execute_batch(&format!("ATTACH DATABASE '{escaped}' AS edb"))
            .context("lazy embed: attach embed.db")?;

        // #391: outer eligibility uses the shared NODE_ELIGIBLE predicate so the
        // lazy-embed path can never diverge from the batch/index paths. The bare
        // exclusion list stays inline in the caller/callee sub-queries only.
        let sql = format!(
            "SELECT f.rowid AS node_id, \
             COALESCE(n.embed_text, \
                 n.kind || ': ' || n.signature \
                 || COALESCE(' | module: ' || NULLIF(n.path, ''), '') \
                 || COALESCE(' | callers: ' || ( \
                     SELECT GROUP_CONCAT(sub.sig, ', ') FROM ( \
                         SELECT SUBSTR(src_n.signature, 1, 60) AS sig \
                         FROM edges e JOIN nodes src_n ON src_n.id = e.src \
                         WHERE e.dst = n.id \
                         AND src_n.kind NOT IN \
                             ('file','file-module','import','module','field','variable') \
                         LIMIT 5) AS sub), '') \
                 || COALESCE(' | callees: ' || ( \
                     SELECT GROUP_CONCAT(sub.sig, ', ') FROM ( \
                         SELECT SUBSTR(dst_n.signature, 1, 60) AS sig \
                         FROM edges e JOIN nodes dst_n ON dst_n.id = e.dst \
                         WHERE e.src = n.id \
                         AND dst_n.kind NOT IN \
                             ('file','file-module','import','module','field','variable') \
                         LIMIT 5) AS sub), '')) AS text \
             FROM nodes_fts f \
             JOIN nodes n ON n.id = f.rowid \
             WHERE nodes_fts MATCH ?1 \
             AND {node_eligible} \
             AND NOT EXISTS ( \
                 SELECT 1 FROM edb.node_embeddings e \
                 WHERE e.node_id = n.id AND e.model_id = ?2 \
             ) \
             ORDER BY rank \
             LIMIT ?3",
            node_eligible = crate::index::NODE_ELIGIBLE,
        );
        let mut stmt = conn
            .prepare(&sql)
            .context("lazy embed: prepare FTS query")?;

        let candidates: Vec<(i64, String)> = stmt
            .query_map(
                rusqlite::params![fts_query, self.model_id.as_str(), limit as i64],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .context("lazy embed: execute FTS query")?
            .filter_map(|r| r.ok())
            .collect();

        Ok(candidates)
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Persist a batch of (node_id, blob) pairs to embed.db after lazy on-the-fly
/// embedding. Called from a background thread — all errors are non-fatal.
fn persist_lazy_embeddings(
    embed_db_path: &Path,
    pairs: &[(i64, Vec<u8>, String)],
    model_id: &str,
) -> Result<()> {
    let conn = Connection::open(embed_db_path).context("lazy persist: open embed.db")?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 30000;",
    )
    .context("lazy persist: configure embed.db")?;
    crate::freshness::ensure_schema(&conn, "").context("lazy persist: ensure schema")?;
    conn.execute("BEGIN", []).context("lazy persist: begin")?;
    let mut ins = conn
        .prepare(
            "INSERT OR IGNORE INTO node_embeddings (node_id, model_id, embedding, text_hash) \
             VALUES (?1, ?2, ?3, ?4)",
        )
        .context("lazy persist: prepare insert")?;
    for (nid, blob, hash) in pairs {
        ins.execute(rusqlite::params![nid, model_id, blob, hash])
            .context("lazy persist: insert")?;
    }
    conn.execute("COMMIT", []).context("lazy persist: commit")?;
    Ok(())
}

/// Read the backend provenance recorded in embed.db meta. None when embed.db,
/// the meta table, or the key don't exist yet (pre-issue-#6 embed.db files).
fn read_backend_provenance(embed_db_path: &Path) -> Option<String> {
    if !embed_db_path.exists() {
        return None;
    }
    let conn = Connection::open_with_flags(
        embed_db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;
    conn.query_row(
        "SELECT value FROM meta WHERE key = 'embed_backend'",
        [],
        |r| r.get(0),
    )
    .ok()
}

/// Record which engine produced this reindex run's vectors in embed.db meta.
///
/// Issue #6: GPU fp32 and tract fp32 matmul are not bit-identical (accumulation
/// order). Negligible for cosine similarity, but a backend change on an
/// incremental reindex makes embed.db mixed-provenance — WARN and recommend a
/// full rebuild instead of silently mixing engines.
fn record_backend_provenance(embed_db_path: &Path, backend_name: &str) -> Result<()> {
    let conn = Connection::open(embed_db_path).context("provenance: open embed.db")?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA busy_timeout = 30000;
         CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    )
    .context("provenance: ensure meta table")?;
    let prev: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'embed_backend'",
            [],
            |r| r.get(0),
        )
        .ok();
    if let Some(prev) = prev.filter(|p| p != backend_name) {
        tracing::warn!(
            previous = %prev,
            current = backend_name,
            "embedding backend changed — existing vectors are from a different engine"
        );
        println!(
            "  WARNING: embedding backend changed ({prev} → {backend_name}). Existing \
             vectors were produced by a different engine; run \
             `travsr embed reindex --rebuild` for consistent embeddings."
        );
    }
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_backend', ?1)",
        [backend_name],
    )
    .context("provenance: write embed_backend")?;
    Ok(())
}

fn write_current_embed_model_meta(conn: &rusqlite::Connection, model_id: &str) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('current_embed_model', ?1)",
        [model_id],
    )
    .context("writing current_embed_model meta")?;
    Ok(())
}

/// Byte-count-based token estimate for bin-packing: BPE encodes ~4 bytes/token
/// for ASCII code. +4 accounts for the "search_document:" prefix tokens.
/// Accurate enough to bound padded tensor cost; actual count differs by <20%.
fn estimate_tokens(text: &str) -> usize {
    text.len() / 4 + 4
}

/// Build the text string fed to the embedding model for one node.
///
/// Enriches the base `kind: signature` with module path and immediate caller /
/// callee names so that private/internal symbols are discoverable via the
/// concepts of their neighbours — not just their own name.
///
/// Caller and callee strings are already comma-joined by the SQL layer (up to 5
/// each, truncated to 60 chars per signature). Either may be `None` when the
/// node has no graph neighbours of the included kinds.
fn build_node_text(
    kind: &str,
    sig: &str,
    path: &str,
    callers: Option<&str>,
    callees: Option<&str>,
) -> String {
    let mut text = format!("{kind}: {sig}");
    if !path.is_empty() {
        text.push_str(" | module: ");
        text.push_str(path);
    }
    if let Some(c) = callers {
        if !c.is_empty() {
            text.push_str(" | callers: ");
            text.push_str(c);
        }
    }
    if let Some(d) = callees {
        if !d.is_empty() {
            text.push_str(" | callees: ");
            text.push_str(d);
        }
    }
    text
}

/// Derive the embed.db path as a sibling of graph.db.
fn embed_db_path_for(db_path: &Path) -> PathBuf {
    db_path.with_file_name("embed.db")
}

// ── --reindex mode ────────────────────────────────────────────────────────────

/// One-shot embedding: read all nodes from graph.db that do not yet have a
/// nomic-v1.5-int8 row in embed.db.node_embeddings, embed them in
/// token-budget-bounded batches, write the BLOBs to embed.db, and update the
/// per-repo HNSW index.
///
/// RFC-019: node_embeddings lives in embed.db (separate from graph.db) to
/// eliminate WAL write contention. graph.db is used read-only for node queries;
/// embed.db is ATTACHed with synchronous=OFF + wal_autocheckpoint=0 for bulk
/// writes (~8-15× faster than writing into the shared graph.db WAL).
///
/// CDC tombstones: node deletions captured in graph.db.node_tombstones are
/// applied to embed.db.node_embeddings atomically before the embedding loop,
/// then acked by clearing the tombstone table. At-least-once delivery: if the
/// sidecar crashes between delete and ack, tombstones replay on next run.
///
/// When `shard = Some((i, n))`, only processes nodes where `id % n = i` and
/// skips all HNSW operations — the CLI orchestrator calls rebuild_index() when
/// all n shards have finished.
#[allow(clippy::too_many_arguments)]
fn reindex(
    model_dir: &Path,
    db_path: &Path,
    embed_db_path: &Path,
    shard: Option<(usize, usize)>,
    row_range: Option<(i64, i64)>,
    busy_timeout_ms: u64,
    phase: Phase,
    model_id: &str,
) -> Result<()> {
    // worker_mode: either shard or range partitioning — skip HNSW per-worker
    let worker_mode = shard.is_some() || row_range.is_some();

    // Partition clause: integer literals are safe (our own values, not user SQL).
    let partition_clause = match (shard, row_range) {
        (_, Some((start, end))) => format!("AND n.id >= {start} AND n.id < {end}"),
        (Some((idx, total)), _) => {
            format!("AND (((n.id % {total}) + {total}) % {total} = {idx})")
        }
        (None, None) => String::new(),
    };

    let worker_label: String = match (shard, row_range) {
        (_, Some((start, end))) => format!("range [{start},{end})"),
        (Some((idx, total)), _) => format!("shard {idx}/{total}"),
        (None, None) => String::new(),
    };

    tracing::info!(
        db = %db_path.display(),
        embed_db = %embed_db_path.display(),
        worker = %worker_label,
        "starting reindex"
    );

    let desc = model::ModelDescriptor::load(model_dir).context("loading model descriptor")?;
    let dim = desc.output_dim();
    let model = backend::create_backend(model_dir, &desc).context("loading model")?;
    record_backend_provenance(embed_db_path, model.backend_name())?;

    // graph.db: node source + tombstone log. synchronous=NORMAL is fine — we
    // only write the tombstone ack and meta, not the bulk embedding BLOBs.
    let conn = Connection::open(db_path).context("open graph.db")?;
    conn.execute_batch(&format!(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA cache_size = -16384;
         PRAGMA busy_timeout = {busy_timeout_ms};",
    ))
    .context("configure graph.db pragmas")?;

    // embed.db: ATTACH as "edb". RFC-019: synchronous=OFF eliminates per-commit
    // fsyncs (safe — a crash means re-embed on next run, not graph corruption).
    // wal_autocheckpoint=0 lets the WAL grow freely during bulk writes; one
    // explicit TRUNCATE checkpoint at the end commits everything in a single fsync.
    let embed_db_str = embed_db_path
        .to_str()
        .context("embed.db path is not valid UTF-8")?;
    conn.execute_batch(&format!(
        "ATTACH DATABASE '{embed_db_str}' AS edb;
         PRAGMA edb.journal_mode = WAL;
         PRAGMA edb.synchronous = OFF;
         PRAGMA edb.wal_autocheckpoint = 0;
         PRAGMA edb.cache_size = -65536;",
    ))
    .context("attach and configure embed.db")?;

    // Create schema in embed.db on first run, and migrate in `text_hash` on an
    // index built before #376 W3 (idempotent).
    crate::freshness::ensure_schema(&conn, "edb.")?;

    // #376 W3: content-hash invalidation replaces the blanket
    // "tombstone ⇒ delete the vector" rule. Workers skip it — the orchestrator's
    // main-thread pass owns invalidation for the whole run.
    let invalidation = if worker_mode {
        crate::freshness::InvalidationReport::default()
    } else {
        crate::freshness::apply_invalidation(&conn, true, true)
            .context("applying content-hash invalidation")?
    };

    let phase_clause = match phase {
        Phase::All => String::new(),
        Phase::Phase1(t) => format!("AND n.shell_number >= {t} "),
        Phase::Phase2(t) => format!("AND n.shell_number < {t} "),
    };

    // NOT EXISTS checks edb.node_embeddings so graph.db WAL is never touched
    // by embedding writes.
    //
    // Columns 3-5 (path, callers, callees) enrich the embedding text so that
    // private/internal functions are reachable via their neighbours' names.
    // Correlated subqueries use the covering indices idx_edges_dst_kind_cov and
    // idx_edges_src_kind_cov — no table scan needed per node.
    // Correlated subqueries use the covering indices idx_edges_dst_kind_cov and
    // idx_edges_src_kind_cov — no table scan needed per node.
    // #391: `node_eligible` is the shared write/index eligibility predicate; the
    // bare `kind_exclude` list stays for the caller/callee sub-queries only.
    let kind_exclude = "'file','file-module','import','module','field','variable'";
    let node_eligible = crate::index::NODE_ELIGIBLE;
    let sql = format!(
        "SELECT n.id, n.kind, n.signature, n.path, \
         n.embed_text, \
         (SELECT GROUP_CONCAT(sub.sig, ', ') FROM \
             (SELECT SUBSTR(src_n.signature, 1, 60) AS sig \
              FROM edges e JOIN nodes src_n ON src_n.id = e.src \
              WHERE e.dst = n.id \
              AND src_n.kind NOT IN ({kind_exclude}) LIMIT 5) AS sub) AS callers, \
         (SELECT GROUP_CONCAT(sub.sig, ', ') FROM \
             (SELECT SUBSTR(dst_n.signature, 1, 60) AS sig \
              FROM edges e JOIN nodes dst_n ON dst_n.id = e.dst \
              WHERE e.src = n.id \
              AND dst_n.kind NOT IN ({kind_exclude}) LIMIT 5) AS sub) AS callees \
         FROM nodes n \
         WHERE {node_eligible} \
         AND NOT EXISTS ( \
             SELECT 1 FROM edb.node_embeddings e \
             WHERE e.node_id = n.id AND e.model_id = ?1 \
         ) {phase_clause}\
         {partition_clause} \
         ORDER BY \
             CASE WHEN n.path LIKE '%_test.%' OR n.path LIKE '%/testing/%' OR n.path LIKE 'test/%' THEN 0 ELSE 1 END DESC, \
             n.shell_number DESC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let pending: Vec<(i64, String)> = stmt
        .query_map([model_id], |row| {
            let id: i64 = row.get(0)?;
            let kind: String = row.get(1)?;
            let sig: String = row.get(2)?;
            let path: String = row.get(3)?;
            let embed_text: Option<String> = row.get(4)?;
            let callers: Option<String> = row.get(5)?;
            let callees: Option<String> = row.get(6)?;
            let text = embed_text.unwrap_or_else(|| {
                build_node_text(&kind, &sig, &path, callers.as_deref(), callees.as_deref())
            });
            Ok((id, text))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let total = pending.len();
    tracing::info!(total, worker = %worker_label, "symbol nodes to embed");

    let mut texts = pending;
    texts.sort_by_key(|(_, text)| estimate_tokens(text));

    let index_path = index_path_for_db(db_path, model_id);

    if total == 0 {
        if worker_mode {
            println!("  {worker_label}: no pending nodes.");
            return Ok(());
        }
        // #376 W2: a pass that only *removed* vectors (a deleted file, an edited
        // chunk whose re-embed is not this phase's job) still has to rebuild the
        // HNSW — otherwise the deleted vector stays live in the index file and
        // keeps being returned by KNN, which is how a deleted doc outlives its
        // file. Checked before the phase/short-circuit returns below.
        if invalidation.removed_any() {
            println!("  Invalidation removed vectors — rebuilding HNSW index...");
            rebuild_index(db_path, embed_db_path, model_id)?;
            return Ok(());
        }
        if matches!(phase, Phase::Phase2(_)) {
            println!("Phase 2 complete — no pending symbol nodes.");
            return Ok(());
        }
        if index_path.exists() {
            write_current_embed_model_meta(&conn, model_id)?;
            println!("All nodes already have embeddings for {model_id}. Index up to date.");
            return Ok(());
        }
        let existing: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM edb.node_embeddings WHERE model_id = ?1",
                [model_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        println!("All nodes already embedded ({existing} rows). Building missing HNSW index...");
        // #376 Phase 2: rebuild_index() builds both the code index (missing here)
        // and, when any exist, the doc-space index — a plain build_from_db call
        // would silently skip doc-chunk vectors. It opens its own connection to
        // db_path; SQLite's WAL mode allows this alongside the still-live `conn`.
        rebuild_index(db_path, embed_db_path, model_id)?;
        return Ok(());
    }

    let idx: Option<index::VecIndex> = if worker_mode || matches!(phase, Phase::Phase2(_)) {
        None
    } else {
        let idx = if index_path.exists() {
            index::VecIndex::try_load(&index_path)
                .context("load existing HNSW index")?
                .expect("index file exists but load returned None")
        } else {
            let existing: usize = conn
                .query_row(
                    "SELECT COUNT(*) FROM edb.node_embeddings WHERE model_id = ?1",
                    [model_id],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            if existing > 0 {
                index::VecIndex::build_from_db(
                    db_path,
                    embed_db_path,
                    model_id,
                    &index_path,
                    existing,
                    dim,
                    crate::index::CODE_SPACE_ELIGIBLE,
                )
                .context("rebuild HNSW from existing embeddings before adding pending")?;
                index::VecIndex::try_load(&index_path)
                    .context("load freshly-rebuilt HNSW")?
                    .expect("just-rebuilt index must be loadable")
            } else {
                index::VecIndex::new_empty(&index_path, total, dim)
                    .context("create new HNSW index")?
            }
        };
        idx.reserve(idx.size() + total)
            .context("reserve HNSW capacity for pending nodes")?;
        Some(idx)
    };

    let est_lens: Vec<usize> = texts
        .iter()
        .map(|(_, text)| estimate_tokens(text))
        .collect();

    let mut batch_ranges: Vec<std::ops::Range<usize>> = Vec::new();
    {
        let mut batch_start = 0usize;
        let mut batch_max_est = 0usize;
        for (i, &est) in est_lens.iter().enumerate() {
            let new_max = batch_max_est.max(est);
            let projected_tokens = new_max * (i - batch_start + 1);
            if i > batch_start
                && (projected_tokens > TOKEN_BUDGET || (i - batch_start) >= MAX_BATCH)
            {
                batch_ranges.push(batch_start..i);
                batch_start = i;
                batch_max_est = est;
            } else {
                batch_max_est = new_max;
            }
        }
        if batch_start < texts.len() {
            batch_ranges.push(batch_start..texts.len());
        }
    }

    // INSERT into edb.node_embeddings — never touches graph.db WAL.
    // #376 W3: `text_hash` records the exact text this vector was built from, so
    // a later pass can tell "re-indexed" from "actually changed".
    let mut ins = conn.prepare(
        "INSERT OR REPLACE INTO edb.node_embeddings (node_id, model_id, embedding, text_hash) \
         VALUES (?1, ?2, ?3, ?4)",
    )?;

    let mut tx_buffer: Vec<(i64, Vec<u8>, String)> = Vec::with_capacity(TX_BATCH + 512);
    let mut inserted = 0usize;
    // #376 O2: see TX_FLUSH_INTERVAL — the row count this drives is what the
    // host's no-progress watchdog observes.
    let mut last_flush = std::time::Instant::now();

    let flush_buffer = |tx_buffer: &Vec<(i64, Vec<u8>, String)>,
                        conn: &rusqlite::Connection,
                        ins: &mut rusqlite::Statement<'_>,
                        idx: &Option<index::VecIndex>|
     -> Result<()> {
        conn.execute("BEGIN", [])?;
        for (node_id, blob, hash) in tx_buffer {
            ins.execute(rusqlite::params![node_id, model_id, blob, hash])?;
            if let Some(ref idx_inner) = idx {
                let vec = model::blob_to_f32(blob);
                idx_inner.add(*node_id, &vec)?;
            }
        }
        conn.execute("COMMIT", [])?;
        Ok(())
    };

    for range in batch_ranges {
        let chunk = &texts[range];
        let text_refs: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();

        let blobs = model.embed_documents(&text_refs)?;

        for ((node_id, text), blob) in chunk.iter().zip(blobs.iter()) {
            tx_buffer.push((*node_id, blob.clone(), crate::freshness::text_hash(text)));
        }

        if should_flush(tx_buffer.len(), last_flush) {
            flush_buffer(&tx_buffer, &conn, &mut ins, &idx)?;
            inserted += tx_buffer.len();
            if inserted % 1_000 < tx_buffer.len() || inserted >= total {
                println!("  embedded {inserted}/{total}");
            }
            tx_buffer.clear();
            last_flush = std::time::Instant::now();
        }
    }

    if !tx_buffer.is_empty() {
        flush_buffer(&tx_buffer, &conn, &mut ins, &idx)?;
        inserted += tx_buffer.len();
        println!("  embedded {inserted}/{total}");
        tx_buffer.clear();
    }

    // Single fsync for all embed.db WAL writes — far cheaper than per-TX fsyncs.
    conn.execute_batch("PRAGMA edb.wal_checkpoint(TRUNCATE)")
        .context("checkpoint embed.db WAL")?;

    if worker_mode {
        println!("  {worker_label}: {inserted} nodes embedded.");
    } else {
        drop(idx);
        let phase_label = if matches!(phase, Phase::Phase2(_)) {
            "Phase 2"
        } else {
            "Phase 1"
        };
        println!("  {phase_label} complete — {inserted} nodes embedded.");
        println!("  Rebuilding HNSW index from all embeddings...");
        // Drop the prepared statement before rebuild_index opens its own connection.
        drop(ins);
        rebuild_index(db_path, embed_db_path, model_id)?;
    }

    tracing::info!(inserted, total, worker = %worker_label, "reindex complete");
    Ok(())
}

// ── --parallel N mode ─────────────────────────────────────────────────────────

/// Parallel reindex: load the model ONCE; N inference threads run concurrently.
///
/// Compared to the old multi-process design (RFC-020), this eliminates the
/// N × 270 MB model-load memory cliff. RAM usage is ~constant regardless of N:
/// 1 × model_weights (~127 MB) + N × per-connection SQLite caches.
///
/// Pipeline (shared atomic batch queue):
///   Main thread materialises ALL pending (id, text) pairs in one read pass,
///   pre-builds ALL inference batches globally (sorted shortest-first), then
///   exposes them via Arc<AtomicUsize> counter.  Workers loop: claim next batch
///   index atomically → embed → write to embed.db.  All N workers run until the
///   queue is empty — no worker idles while others still have pre-assigned work.
///   Bottleneck is AMX saturation (4 threads share one AMX unit), so scheduling
///   gains are marginal; this form is kept for correctness and code clarity.
///   WAL serialises concurrent COMMITs; inference dominates write contention.
#[allow(clippy::too_many_arguments)]
fn reindex_parallel(
    model_dir: &Path,
    db_path: &Path,
    embed_db_path: &Path,
    parallel: usize,
    busy_timeout_ms: u64,
    phase: Phase,
    model_id: &str,
    // WS3: when set, a watcher thread polls this path; on appearance it drains all
    // workers (finishing their in-flight batch) so the daemon/CLI can cancel a
    // running reindex within one batch. Embedded rows are preserved incrementally.
    cancel_sentinel: Option<&Path>,
    // WS3 §4.3: on cancel, skip the end-of-run HNSW rebuild by default (fast stop).
    // `--rebuild-on-cancel` forces the atomic rebuild so the partial set is queryable.
    rebuild_on_cancel: bool,
) -> Result<()> {
    tracing::info!(
        parallel,
        db = %db_path.display(),
        embed_db = %embed_db_path.display(),
        "parallel reindex: {} reader thread(s), single model load",
        parallel
    );

    // ── Step 1: invalidation (main thread, needs write access to both dbs) ────
    // #376 W3: content-hash verification, not a blanket tombstone delete. See
    // `freshness` for why the two spaces are verified differently.
    let invalidation = {
        let conn = Connection::open(db_path).context("open graph.db for invalidation")?;
        let embed_str = embed_db_path
            .to_str()
            .context("embed.db path is not valid UTF-8")?;
        let escaped = embed_str.replace('\'', "''");
        conn.execute_batch(&format!(
            "PRAGMA busy_timeout = {busy_timeout_ms};
             ATTACH DATABASE '{escaped}' AS edb;
             PRAGMA edb.journal_mode = WAL;"
        ))
        .context("configure connections for invalidation pass")?;
        crate::freshness::ensure_schema(&conn, "edb.")?;
        crate::freshness::apply_invalidation(&conn, true, true)
            .context("applying content-hash invalidation")?
    };

    // ── Step 2: materialise ALL pending (id, text) pairs on the main thread ──
    // One read pass with NOT EXISTS applied here — workers receive their chunk
    // by value and need no SQL connection of their own.
    let phase_clause = match phase {
        Phase::All => String::new(),
        Phase::Phase1(t) => format!("AND n.shell_number >= {t}"),
        Phase::Phase2(t) => format!("AND n.shell_number < {t}"),
    };

    let all_pending: Vec<(i64, String)> = {
        let conn = Connection::open_with_flags(
            db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("open graph.db for pending query")?;
        let embed_str = embed_db_path
            .to_str()
            .context("embed.db path is not valid UTF-8")?;
        let escaped = embed_str.replace('\'', "''");
        conn.execute_batch(&format!("ATTACH DATABASE '{escaped}' AS edb"))
            .context("attach embed.db for pending query")?;
        // #391: shared eligibility predicate (see NODE_ELIGIBLE); the bare
        // `kind_exclude` list remains for the caller/callee sub-queries only.
        let kind_exclude = "'file','file-module','import','module','field','variable'";
        let node_eligible = crate::index::NODE_ELIGIBLE;
        let sql = format!(
            "SELECT n.id, n.kind, n.signature, n.path, \
             n.embed_text, \
             (SELECT GROUP_CONCAT(sub.sig, ', ') FROM \
                 (SELECT SUBSTR(src_n.signature, 1, 60) AS sig \
                  FROM edges e JOIN nodes src_n ON src_n.id = e.src \
                  WHERE e.dst = n.id \
                  AND src_n.kind NOT IN ({kind_exclude}) LIMIT 5) AS sub) AS callers, \
             (SELECT GROUP_CONCAT(sub.sig, ', ') FROM \
                 (SELECT SUBSTR(dst_n.signature, 1, 60) AS sig \
                  FROM edges e JOIN nodes dst_n ON dst_n.id = e.dst \
                  WHERE e.src = n.id \
                  AND dst_n.kind NOT IN ({kind_exclude}) LIMIT 5) AS sub) AS callees \
             FROM nodes n \
             WHERE {node_eligible} \
             AND NOT EXISTS (\
                 SELECT 1 FROM edb.node_embeddings e \
                 WHERE e.node_id = n.id AND e.model_id = ?1\
             ) {phase_clause} \
             ORDER BY n.shell_number DESC"
        );
        let mut stmt = conn.prepare(&sql).context("prepare pending query")?;
        let rows: Vec<(i64, String)> = stmt
            .query_map([model_id], |row| {
                let id: i64 = row.get(0)?;
                let kind: String = row.get(1)?;
                let sig: String = row.get(2)?;
                let path: String = row.get(3)?;
                let embed_text: Option<String> = row.get(4)?;
                let callers: Option<String> = row.get(5)?;
                let callees: Option<String> = row.get(6)?;
                let text = embed_text.unwrap_or_else(|| {
                    build_node_text(&kind, &sig, &path, callers.as_deref(), callees.as_deref())
                });
                Ok((id, text))
            })
            .context("query pending nodes")?
            .filter_map(|r| r.ok())
            .collect();
        rows
    };

    let total = all_pending.len();
    if total == 0 {
        // #376 W2: nothing to embed, but if invalidation removed vectors the
        // live HNSW still contains them. Deletions must reach the index even
        // when no inference is needed.
        if invalidation.removed_any() {
            println!("  Invalidation removed vectors — rebuilding HNSW index...");
            rebuild_index(db_path, embed_db_path, model_id)?;
            return Ok(());
        }
        println!("All nodes already embedded — nothing to do.");
        return Ok(());
    }

    // ── Step 3: shared atomic batch queue (Kafka-style consumer group) ────────
    // Sort all items shortest-first (optimal BatchLongest padding), pre-build
    // ALL inference batches globally, then expose them via a shared AtomicUsize
    // counter. Workers loop: atomically claim the next batch index → embed →
    // write. All N workers run until the queue is empty; no worker idles while
    // another still has pre-assigned work. Workload is AMX-bound in practice,
    // so scheduling gains are marginal (~3%); this form is kept for code clarity.
    let mut sorted = all_pending;
    sorted.sort_by_key(|(_, t)| estimate_tokens(t));
    let est_lens: Vec<usize> = sorted.iter().map(|(_, t)| estimate_tokens(t)).collect();
    let batch_ranges = build_batch_ranges(&est_lens);
    let n_batches = batch_ranges.len();

    let items = Arc::new(sorted);
    let batches = Arc::new(batch_ranges);
    let next_batch = Arc::new(AtomicUsize::new(0));

    // WS3: cancel watcher. Polls the sentinel every 250ms; on appearance it stores
    // `next_batch = n_batches` so every worker's next `fetch_add` exceeds the queue
    // length → each finishes its in-flight batch, commits, and breaks. Portable
    // (no signals), works with or without the daemon. `watcher_done` stops the
    // thread once the workers finish normally.
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watcher_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watcher = cancel_sentinel.map(|sentinel| {
        let sentinel = sentinel.to_path_buf();
        let next_c = Arc::clone(&next_batch);
        let cancelled_c = Arc::clone(&cancelled);
        let done_c = Arc::clone(&watcher_done);
        std::thread::Builder::new()
            .name("embed-cancel-watch".into())
            .spawn(move || {
                while !done_c.load(Ordering::Relaxed) {
                    if sentinel.exists() {
                        tracing::info!("cancel sentinel detected — draining workers");
                        next_c.store(n_batches, Ordering::Relaxed);
                        cancelled_c.store(true, Ordering::SeqCst);
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
            })
            .expect("spawn cancel watcher")
    });

    let desc = model::ModelDescriptor::load(model_dir).context("loading model descriptor")?;
    let model = backend::create_backend(model_dir, &desc).context("loading model")?;
    record_backend_provenance(embed_db_path, model.backend_name())?;
    // Issue #6: N CPU worker threads do not map onto one GPU. When the backend
    // is accelerated, run a single submit loop — the backend pads each batch to
    // fixed shape buckets (avoids CUDA/TensorRT per-shape re-optimisation) and
    // `--parallel` stays a CPU-only hint. WS3 cancel-sentinel is unaffected:
    // it drains via the shared batch counter either way.
    let n_workers = if model.is_accelerated() {
        if parallel > 1 {
            tracing::info!(
                requested = parallel,
                backend = model.backend_name(),
                "accelerated backend — --parallel is a CPU-only hint; using a single submit loop"
            );
        }
        1
    } else {
        parallel.min(n_batches).max(1)
    };
    tracing::info!(
        total,
        n_batches,
        n_workers,
        "model loaded; spawning {} inference threads (shared batch queue)",
        n_workers
    );

    // ── Steps 4-6: N consumer threads ────────────────────────────────────────
    // Workers share Arc<Vec<items>>, Arc<Vec<ranges>>, Arc<AtomicUsize>.
    // The backend is shared via Arc<dyn EmbedBackend> — one model load total.
    // Each worker opens its own write connection to embed.db; WAL serialises COMMITs.
    let edb_arc = Arc::new(embed_db_path.to_path_buf());
    let mid_arc = Arc::new(model_id.to_owned());

    let worker_handles: Vec<_> = (0..n_workers)
        .map(|i| {
            let model_w = model.clone();
            let edb_w = Arc::clone(&edb_arc);
            let mid_w = Arc::clone(&mid_arc);
            let items_w = Arc::clone(&items);
            let batches_w = Arc::clone(&batches);
            let next_w = Arc::clone(&next_batch);

            std::thread::Builder::new()
                .name(format!("embed-{i}"))
                .spawn(move || -> Result<usize> {
                    // ── write: own connection, synchronous=OFF for bulk speed ──
                    let wconn = Connection::open(&*edb_w).context("worker: open embed.db")?;
                    wconn
                        .execute_batch(&format!(
                            "PRAGMA journal_mode = WAL;
                             PRAGMA synchronous = OFF;
                             PRAGMA wal_autocheckpoint = 0;
                             PRAGMA cache_size = -32768;
                             PRAGMA busy_timeout = {busy_timeout_ms};"
                        ))
                        .context("worker: configure embed.db")?;
                    crate::freshness::ensure_schema(&wconn, "").context("worker: ensure schema")?;
                    let mut ins = wconn
                        .prepare(
                            "INSERT OR REPLACE INTO node_embeddings \
                             (node_id, model_id, embedding, text_hash) VALUES (?1, ?2, ?3, ?4)",
                        )
                        .context("worker: prepare insert")?;

                    let mut tx_buf: Vec<(i64, Vec<u8>, String)> =
                        Vec::with_capacity(TX_BATCH + 512);
                    let mut inserted = 0usize;
                    // #376 O2: see TX_FLUSH_INTERVAL.
                    let mut last_flush = std::time::Instant::now();

                    // ── consumer loop: claim batches until the queue is empty ─
                    loop {
                        let batch_idx = next_w.fetch_add(1, Ordering::Relaxed);
                        if batch_idx >= batches_w.len() {
                            break;
                        }
                        let range = batches_w[batch_idx].clone();
                        let batch = &items_w[range];
                        let texts: Vec<&str> = batch.iter().map(|(_, t)| t.as_str()).collect();
                        let blobs = model_w.embed_documents(&texts).context("worker: embed")?;
                        for ((nid, text), blob) in batch.iter().zip(blobs.iter()) {
                            tx_buf.push((*nid, blob.clone(), crate::freshness::text_hash(text)));
                        }
                        if should_flush(tx_buf.len(), last_flush) {
                            wconn.execute("BEGIN", []).context("worker: begin")?;
                            for (nid, blob, hash) in &tx_buf {
                                ins.execute(rusqlite::params![nid, mid_w.as_str(), blob, hash])
                                    .context("worker: insert")?;
                            }
                            wconn.execute("COMMIT", []).context("worker: commit")?;
                            inserted += tx_buf.len();
                            tx_buf.clear();
                            last_flush = std::time::Instant::now();
                        }
                    }

                    if !tx_buf.is_empty() {
                        wconn.execute("BEGIN", []).context("worker: begin final")?;
                        for (nid, blob, hash) in &tx_buf {
                            ins.execute(rusqlite::params![nid, mid_w.as_str(), blob, hash])
                                .context("worker: insert final")?;
                        }
                        wconn
                            .execute("COMMIT", [])
                            .context("worker: commit final")?;
                        inserted += tx_buf.len();
                    }

                    tracing::debug!(thread = i, inserted, "inference worker complete");
                    Ok(inserted)
                })
                .expect("spawn inference thread")
        })
        .collect();

    let mut total_embedded = 0usize;
    let worker_errors: Vec<String> = worker_handles
        .into_iter()
        .enumerate()
        .filter_map(|(i, h)| match h.join() {
            Ok(Ok(n)) => {
                total_embedded += n;
                None
            }
            Ok(Err(e)) => Some(format!("worker {i}: {e:#}")),
            Err(_) => Some(format!("worker {i}: panicked")),
        })
        .collect();
    // Stop the cancel watcher and observe whether a cancel fired.
    watcher_done.store(true, Ordering::Relaxed);
    if let Some(w) = watcher {
        let _ = w.join();
    }
    let was_cancelled = cancelled.load(Ordering::SeqCst);
    if !worker_errors.is_empty() {
        anyhow::bail!("inference worker errors:\n  {}", worker_errors.join("\n  "));
    }

    println!("  Embedded {total_embedded} nodes.");

    // ── Step 7: single checkpoint across all workers' WAL writes ─────────────
    let write_conn = Connection::open(embed_db_path).context("open embed.db for checkpoint")?;
    write_conn
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .context("checkpoint embed.db WAL")?;
    drop(write_conn);

    // WS3 §4.3: on graceful cancel, fast-stop by default — the embedded rows are
    // durable (per-batch commits) but the live HNSW stays the pre-cancel index
    // until the next *completed* reindex, unless the user opted into rebuild.
    if was_cancelled {
        // Best-effort remove the sentinel so a later `reindex` isn't cancelled on
        // sight. The daemon/CLI polls process exit, not the sentinel, so removing
        // it here does not race the terminate path.
        if let Some(s) = cancel_sentinel {
            let _ = std::fs::remove_file(s);
        }
        if rebuild_on_cancel {
            println!("  Cancelled — rebuilding HNSW over {total_embedded} partial embeddings...");
            rebuild_index(db_path, embed_db_path, model_id)?;
            tracing::info!(total_embedded, "parallel reindex cancelled (rebuilt)");
        } else {
            println!(
                "  Cancelled — {total_embedded} partial embeddings preserved. \
                 Run `travsr embed reindex` to resume and make them searchable."
            );
            tracing::info!(total_embedded, "parallel reindex cancelled (fast-stop)");
        }
        return Ok(());
    }

    if matches!(phase, Phase::Phase2(_)) {
        println!("Phase 2 complete — {total_embedded} nodes embedded.");
    }
    // Always rebuild after Phase 2 so HNSW covers Phase 1 + Phase 2 nodes.
    // Phase 1 rebuilds unconditionally too (same branch).
    println!("  Rebuilding HNSW index from all embeddings...");
    rebuild_index(db_path, embed_db_path, model_id)?;

    tracing::info!(total_embedded, parallel, "parallel reindex complete");
    Ok(())
}

/// Partition a slice of per-item token estimates into token-budget batches.
/// Items must be pre-sorted shortest-first so BatchLongest padding is minimised.
fn build_batch_ranges(est_lens: &[usize]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut max_est = 0usize;
    for (i, &est) in est_lens.iter().enumerate() {
        let new_max = max_est.max(est);
        let projected = new_max * (i - start + 1);
        if i > start && (projected > TOKEN_BUDGET || (i - start) >= MAX_BATCH) {
            ranges.push(start..i);
            start = i;
            max_est = est;
        } else {
            max_est = new_max;
        }
    }
    if start < est_lens.len() {
        ranges.push(start..est_lens.len());
    }
    ranges
}

// ── --rebuild-index mode ──────────────────────────────────────────────────────

/// Rebuild the per-repo HNSW index by streaming all rows from embed.db.node_embeddings.
/// No ONNX inference — pure SQLite I/O.  Used as the final step by the CLI
/// orchestrator after parallel shard embedding completes.
fn rebuild_index(db_path: &Path, embed_db_path: &Path, model_id: &str) -> Result<()> {
    tracing::info!(
        db = %db_path.display(),
        embed_db = %embed_db_path.display(),
        "rebuilding HNSW index"
    );
    // Dim comes from the model descriptor (resolved from the model dir) — never
    // hardcoded. rebuild_index has no model_dir arg, so resolve it from model_id.
    let dim = model::ModelDescriptor::load(&model_dir(model_id)?)?.output_dim();
    let conn = Connection::open(db_path).context("open graph.db")?;
    let embed_db_str = embed_db_path
        .to_str()
        .context("embed.db path is not valid UTF-8")?;
    conn.execute_batch(&format!("ATTACH DATABASE '{embed_db_str}' AS edb"))
        .context("attach embed.db")?;

    // #376 W2: this is the one path that writes an HNSW file without embedding
    // anything, and before W2 it was also the one path that could publish an
    // index built from vectors already known to be stale. Verify first. `ack`
    // is false on purpose: this path can *remove* a stale vector but cannot
    // re-embed it, so the tombstone must survive for the next real pass.
    match crate::freshness::apply_invalidation(&conn, false, true) {
        Ok(r) if r.removed_any() => {
            tracing::info!(
                stale = r.stale,
                orphaned = r.orphaned,
                doc_stale = r.doc_stale,
                "rebuild-index: dropped stale vectors before building"
            );
        }
        Ok(_) => {}
        // A rebuild on a db with no tombstone table (or a read-only graph.db)
        // must still produce an index — freshness is best-effort here.
        Err(e) => tracing::warn!("rebuild-index: invalidation skipped (non-fatal): {e:#}"),
    }

    // #376 Phase 2: code and doc spaces are counted and built independently —
    // CODE_SPACE_ELIGIBLE and DOC_SPACE_ELIGIBLE partition the corpus, and a
    // repo may legitimately have embeddings in only one (no markdown, or
    // `docs.enabled` never turned on at index time).
    let code_existing: usize = conn
        .query_row(
            &format!(
                "SELECT COUNT(*) FROM edb.node_embeddings e \
                 JOIN nodes n ON n.id = e.node_id \
                 WHERE e.model_id = ?1 AND {}",
                crate::index::CODE_SPACE_ELIGIBLE
            ),
            [model_id],
            |r| r.get(0),
        )
        .context("counting existing code embeddings")?;
    let doc_existing: usize = conn
        .query_row(
            &format!(
                "SELECT COUNT(*) FROM edb.node_embeddings e \
                 JOIN nodes n ON n.id = e.node_id \
                 WHERE e.model_id = ?1 AND {}",
                crate::index::DOC_SPACE_ELIGIBLE
            ),
            [model_id],
            |r| r.get(0),
        )
        .unwrap_or(0);
    anyhow::ensure!(
        code_existing > 0 || doc_existing > 0,
        "no embeddings in embed.db — run `travsr embed reindex` first"
    );

    if code_existing > 0 {
        let index_path = index_path_for_db(db_path, model_id);
        println!("Building code HNSW index from {code_existing} embeddings...");
        index::VecIndex::build_from_db(
            db_path,
            embed_db_path,
            model_id,
            &index_path,
            code_existing,
            dim,
            crate::index::CODE_SPACE_ELIGIBLE,
        )
        .context("build_from_db (code space)")?;
        println!("Done — code index saved to {}.", index_path.display());
    }
    if doc_existing > 0 {
        let doc_index_path = doc_index_path_for_db(db_path, model_id);
        println!("Building doc-space HNSW index from {doc_existing} embeddings...");
        index::VecIndex::build_from_db(
            db_path,
            embed_db_path,
            model_id,
            &doc_index_path,
            doc_existing,
            dim,
            crate::index::DOC_SPACE_ELIGIBLE,
        )
        .context("build_from_db (doc space)")?;
        println!("Done — doc index saved to {}.", doc_index_path.display());
    }

    write_current_embed_model_meta(&conn, model_id)?;
    tracing::info!(code_existing, doc_existing, "HNSW index rebuilt");
    Ok(())
}

// ── entry point ───────────────────────────────────────────────────────────────

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    // WS3 capability probe: `travsr-embed --version` prints the crate version and
    // exits 0. The travsr daemon runs this once to decide whether the sidecar
    // understands `--cancel-sentinel`; an old sidecar without this flag exits
    // non-zero on the unknown arg, so travsr falls back to force-kill on cancel.
    if args.iter().skip(1).any(|a| a == "--version" || a == "-V") {
        println!("travsr-embed {}", env!("CARGO_PKG_VERSION"));
        std::process::exit(0);
    }

    let mut reindex_db: Option<PathBuf> = None;
    let mut daemon_db: Option<PathBuf> = None;
    let mut rebuild_db: Option<PathBuf> = None;
    let mut embed_db: Option<PathBuf> = None;
    let mut shard: Option<(usize, usize)> = None;
    let mut row_start: Option<i64> = None;
    let mut row_end: Option<i64> = None;
    let mut parallel: Option<usize> = None;
    let mut busy_timeout_ms: u64 = 120_000;
    let mut phase = Phase::All;
    let mut model_id_arg: Option<String> = None;
    let mut cancel_sentinel: Option<PathBuf> = None;
    let mut rebuild_on_cancel = false;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--phase1" => {
                i += 1;
                let t = args
                    .get(i)
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or_else(|| {
                        eprintln!("usage: --phase1 <shell-threshold>");
                        std::process::exit(1);
                    });
                phase = Phase::Phase1(t);
            }
            "--phase2" => {
                i += 1;
                let t = args
                    .get(i)
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or_else(|| {
                        eprintln!("usage: --phase2 <shell-threshold>");
                        std::process::exit(1);
                    });
                phase = Phase::Phase2(t);
            }
            "--reindex" => {
                i += 1;
                reindex_db = Some(args.get(i).map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("usage: travsr-embed-nomic --reindex <graph.db-path>");
                    std::process::exit(1);
                }));
            }
            "--embed-db" => {
                i += 1;
                embed_db = Some(args.get(i).map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("usage: --embed-db <embed.db-path>");
                    std::process::exit(1);
                }));
            }
            "--db-path" => {
                i += 1;
                daemon_db = Some(args.get(i).map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("usage: travsr-embed-nomic --db-path <graph.db-path>");
                    std::process::exit(1);
                }));
            }
            "--rebuild-index" => {
                i += 1;
                rebuild_db = Some(args.get(i).map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("usage: travsr-embed-nomic --rebuild-index <graph.db-path>");
                    std::process::exit(1);
                }));
            }
            "--shard" => {
                i += 1;
                let spec = args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("usage: --shard <idx>/<total>");
                    std::process::exit(1);
                });
                let parts: Vec<&str> = spec.splitn(2, '/').collect();
                if parts.len() != 2 {
                    eprintln!("--shard requires <idx>/<total> format, e.g. --shard 0/4");
                    std::process::exit(1);
                }
                let shard_idx = parts[0].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("--shard index must be a non-negative integer");
                    std::process::exit(1);
                });
                let n_shards = parts[1].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("--shard total must be a positive integer");
                    std::process::exit(1);
                });
                if n_shards == 0 || shard_idx >= n_shards {
                    eprintln!("--shard: index must be < total and total must be > 0");
                    std::process::exit(1);
                }
                shard = Some((shard_idx, n_shards));
            }
            "--row-start" => {
                i += 1;
                row_start = Some(
                    args.get(i)
                        .and_then(|s| s.parse::<i64>().ok())
                        .unwrap_or_else(|| {
                            eprintln!("usage: --row-start <i64>");
                            std::process::exit(1);
                        }),
                );
            }
            "--row-end" => {
                i += 1;
                row_end = Some(
                    args.get(i)
                        .and_then(|s| s.parse::<i64>().ok())
                        .unwrap_or_else(|| {
                            eprintln!("usage: --row-end <i64>");
                            std::process::exit(1);
                        }),
                );
            }
            "--busy-timeout-ms" => {
                i += 1;
                busy_timeout_ms = args
                    .get(i)
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or_else(|| {
                        eprintln!("usage: --busy-timeout-ms <ms>");
                        std::process::exit(1);
                    });
            }
            "--parallel" => {
                i += 1;
                let n = args
                    .get(i)
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or_else(|| {
                        eprintln!("usage: --parallel <N>  (N >= 1)");
                        std::process::exit(1);
                    });
                if n == 0 {
                    eprintln!("--parallel N must be >= 1");
                    std::process::exit(1);
                }
                parallel = Some(n);
            }
            "--model-id" => {
                i += 1;
                model_id_arg = Some(args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("usage: --model-id <id>");
                    std::process::exit(1);
                }));
            }
            "--cancel-sentinel" => {
                i += 1;
                cancel_sentinel = Some(args.get(i).map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("usage: --cancel-sentinel <path>");
                    std::process::exit(1);
                }));
            }
            "--rebuild-on-cancel" => {
                rebuild_on_cancel = true;
            }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!(
                    "usage: travsr-embed-nomic \
                     [--model-id <id>] \
                     [--reindex <db> [--embed-db <embed.db>] \
                      [--row-start <i64> --row-end <i64>] \
                      [--phase1 <n>|--phase2 <n>] [--shard <i>/<n>] \
                      [--parallel <N> [--cancel-sentinel <path>] [--rebuild-on-cancel]] \
                      [--busy-timeout-ms <ms>]] \
                     [--rebuild-index <db> [--embed-db <embed.db>]] \
                     [--db-path <db>]"
                );
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if shard.is_some() && reindex_db.is_none() {
        eprintln!("--shard requires --reindex");
        std::process::exit(1);
    }
    match (row_start, row_end) {
        (Some(s), Some(e)) if s >= e => {
            eprintln!("--row-start must be less than --row-end");
            std::process::exit(1);
        }
        (Some(_), None) | (None, Some(_)) => {
            eprintln!("--row-start and --row-end must be used together");
            std::process::exit(1);
        }
        _ => {}
    }
    if row_start.is_some() && reindex_db.is_none() {
        eprintln!("--row-start/--row-end requires --reindex");
        std::process::exit(1);
    }
    if row_start.is_some() && shard.is_some() {
        eprintln!("--row-start/--row-end and --shard are mutually exclusive");
        std::process::exit(1);
    }
    if parallel.is_some() && (shard.is_some() || row_start.is_some()) {
        eprintln!("--parallel is mutually exclusive with --shard and --row-start/--row-end");
        std::process::exit(1);
    }
    if parallel.is_some() && reindex_db.is_none() {
        eprintln!("--parallel requires --reindex");
        std::process::exit(1);
    }
    // WS3: cancel is only wired into the parallel worker loop (shared batch queue).
    if (cancel_sentinel.is_some() || rebuild_on_cancel) && parallel.is_none() {
        eprintln!("--cancel-sentinel/--rebuild-on-cancel require --parallel");
        std::process::exit(1);
    }
    if rebuild_on_cancel && cancel_sentinel.is_none() {
        eprintln!("--rebuild-on-cancel requires --cancel-sentinel");
        std::process::exit(1);
    }
    if rebuild_db.is_some() && reindex_db.is_some() {
        eprintln!("--rebuild-index and --reindex are mutually exclusive");
        std::process::exit(1);
    }

    // Resolve model ID: CLI arg → default.
    let model_id = model_id_arg.as_deref().unwrap_or("bge-small-en-v1.5");

    let model_dir = match model_dir(model_id) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("travsr-embed: cannot find model dir: {e:#}");
            std::process::exit(1);
        }
    };

    if let Some(db_path) = rebuild_db {
        let embed_path = embed_db.unwrap_or_else(|| embed_db_path_for(&db_path));
        if let Err(e) = rebuild_index(&db_path, &embed_path, model_id) {
            eprintln!("rebuild-index failed: {e:#}");
            std::process::exit(1);
        }
    } else if let Some(db_path) = reindex_db {
        let embed_path = embed_db.unwrap_or_else(|| embed_db_path_for(&db_path));
        let result = if let Some(n) = parallel {
            // RFC-021: single model loaded once; N reader threads inside the sidecar.
            reindex_parallel(
                &model_dir,
                &db_path,
                &embed_path,
                n,
                busy_timeout_ms,
                phase,
                model_id,
                cancel_sentinel.as_deref(),
                rebuild_on_cancel,
            )
        } else {
            let row_range = row_start.zip(row_end);
            reindex(
                &model_dir,
                &db_path,
                &embed_path,
                shard,
                row_range,
                busy_timeout_ms,
                phase,
                model_id,
            )
        };
        if let Err(e) = result {
            eprintln!("reindex failed: {e:#}");
            std::process::exit(1);
        }
    } else {
        // Daemon / IPC mode. --db-path is required so we know which per-repo
        // HNSW index to load. KNN is served from the in-memory HNSW — req.db_path
        // (now pointing to embed.db per RFC-019) is unused in this mode.
        let db_path = daemon_db.unwrap_or_else(|| {
            eprintln!("travsr-embed: --db-path <graph.db> is required in daemon mode");
            eprintln!("  (the travsr daemon passes this automatically)");
            std::process::exit(1);
        });
        let index_path = index_path_for_db(&db_path, model_id);
        match NomicPlugin::load(&model_dir, index_path, db_path.clone(), model_id) {
            Ok(plugin) => {
                tracing::info!(
                    model_dir = %model_dir.display(),
                    model_id  = model_id,
                    db        = %db_path.display(),
                    "embed sidecar ready"
                );
                run_embed_plugin(plugin);
            }
            Err(e) => {
                eprintln!("travsr-embed: startup failed: {e:#}");
                std::process::exit(1);
            }
        }
    }
}

/// Per-repo HNSW index path, co-located with graph.db, keyed by model_id.
fn index_path_for_db(db_path: &Path, model_id: &str) -> PathBuf {
    let dir = db_path.parent().unwrap_or(db_path);
    dir.join(format!("{model_id}.hnsw.usearch"))
}

/// #376 Phase 2: per-repo doc-space HNSW index path, co-located with graph.db,
/// keyed by model_id like [`index_path_for_db`]. Named `-docs.hnsw.usearch`
/// rather than the plan's flat `hnsw-docs.usearch` so it stays keyed per
/// model_id the same way the code index is — a flat name would collide or go
/// stale across a model switch.
fn doc_index_path_for_db(db_path: &Path, model_id: &str) -> PathBuf {
    let dir = db_path.parent().unwrap_or(db_path);
    dir.join(format!("{model_id}-docs.hnsw.usearch"))
}

fn model_dir(model_id: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("HOME not set"))?;
    let dir = home.join(".travsr").join("models").join(model_id);
    anyhow::ensure!(
        dir.exists(),
        "model directory not found: {}\n  Run: travsr embed init --backend {model_id}",
        dir.display()
    );
    Ok(dir)
}

#[cfg(test)]
mod flush_policy_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// The pre-#376-O2 behaviour, stated as a test so a revert is loud: a
    /// sub-`TX_BATCH` buffer that has just been committed does not flush again
    /// on size alone. This is the condition that used to hold for a whole pass.
    #[test]
    fn small_buffer_does_not_flush_on_size() {
        assert!(!should_flush(TX_BATCH - 1, Instant::now()));
        assert!(!should_flush(1, Instant::now()));
    }

    /// A full buffer flushes immediately regardless of age — the original
    /// trigger, unchanged.
    #[test]
    fn full_buffer_flushes_on_size() {
        assert!(should_flush(TX_BATCH, Instant::now()));
        assert!(should_flush(TX_BATCH + 1, Instant::now()));
    }

    /// The actual fix, and the property the host's no-progress watchdog depends
    /// on: **any** non-empty buffer becomes observable once it is
    /// `TX_FLUSH_INTERVAL` old. Asserted at an age past the interval rather
    /// than by sleeping, so the test is instant and deterministic.
    #[test]
    fn any_nonempty_buffer_flushes_once_it_is_old_enough() {
        let old = Instant::now() - (TX_FLUSH_INTERVAL + Duration::from_secs(1));
        assert!(should_flush(1, old), "a single buffered row must commit");
        assert!(should_flush(TX_BATCH - 1, old));
    }

    /// The margin under the watchdog is what makes the fix sufficient, not just
    /// directionally right: `NO_PROGRESS_SECS` is 600 in
    /// `travsr_plugin_host::embed_catalog`, so the interval must leave room for
    /// several flushes inside one watchdog window. Guards against someone
    /// raising `TX_FLUSH_INTERVAL` to a value that reintroduces the kill.
    #[test]
    fn flush_interval_leaves_margin_under_the_host_watchdog() {
        const HOST_NO_PROGRESS_SECS: u64 = 600;
        assert!(
            TX_FLUSH_INTERVAL.as_secs() * 4 <= HOST_NO_PROGRESS_SECS,
            "TX_FLUSH_INTERVAL ({}s) must stay well under the host's \
             NO_PROGRESS_SECS ({HOST_NO_PROGRESS_SECS}s)",
            TX_FLUSH_INTERVAL.as_secs()
        );
    }

    /// An empty buffer must never flush: an empty commit resets the flush timer
    /// without moving the row count the watchdog reads, which would recreate the
    /// original failure with extra steps.
    #[test]
    fn empty_buffer_never_flushes() {
        let old = Instant::now() - (TX_FLUSH_INTERVAL + Duration::from_secs(60));
        assert!(!should_flush(0, old));
        assert!(!should_flush(0, Instant::now()));
    }
}

#[cfg(test)]
mod query_memo_tests {
    use super::*;
    use std::time::Duration;

    fn entry(text: &str) -> QueryEmbedCache {
        QueryEmbedCache {
            text: text.to_string(),
            blob: vec![0u8; 4],
            at: std::time::Instant::now(),
        }
    }

    /// The regression guard for the k8s Gate 4 failure (#376 §7 gate 4).
    ///
    /// The two KNN round trips for one query are separated by the code lane's
    /// cross-encoder rerank. On kubernetes that gap was measured at 6.1-7.1 s,
    /// so the old 5 s TTL expired mid-query and the docs lane re-embedded the
    /// same text — two inferences where the contract allows one. This asserts
    /// the memo now spans a gap the old bound could not.
    #[test]
    fn memo_spans_a_gap_that_the_old_five_second_ttl_could_not() {
        let e = entry("why does the apiserver use a watch cache");
        assert!(
            e.serves(
                "why does the apiserver use a watch cache",
                Duration::from_secs(7)
            ),
            "a 7s gap (measured on kubernetes) must still hit the memo"
        );
    }

    #[test]
    fn memo_expires_eventually() {
        let e = entry("q");
        assert!(e.serves("q", Duration::from_secs(59)));
        assert!(!e.serves("q", QUERY_EMBED_CACHE_TTL));
        assert!(!e.serves("q", Duration::from_secs(3600)));
    }

    /// Keyed on exact text: the memo exists to share ONE query's embedding
    /// across both spaces, never to answer a different query.
    #[test]
    fn memo_never_serves_a_different_query() {
        let e = entry("watch cache");
        assert!(!e.serves("watch  cache", Duration::from_millis(1)));
        assert!(!e.serves("Watch cache", Duration::from_millis(1)));
        assert!(!e.serves("", Duration::from_millis(1)));
    }
}
