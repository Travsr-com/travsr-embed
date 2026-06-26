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
use travsr_plugin_protocol::{EmbedPlugin, EmbedRequest, EmbedResponse, KnnRequest, KnnResponse};
use travsr_plugin_sdk::run_embed_plugin;

/// Embedding dimension for each supported BGE model variant.
fn dim_for_model(model_id: &str) -> usize {
    match model_id {
        "bge-base-en-v1.5" => 768,
        "bge-large-en-v1.5" => 1024,
        _ => 384, // bge-small-en-v1.5 and any future 384-dim model
    }
}

/// Human-readable backend label shown in `travsr embed status` and sidecar logs.
fn backend_label(model_id: &str) -> String {
    let dim = dim_for_model(model_id);
    format!("{model_id} fp32 CLS-{dim}")
}
// MAX_BATCH: hard cap on items per forward pass.
// TOKEN_BUDGET: soft cap on padded tensor cost (BatchLongest pads all items to
// max_seq_in_batch; cost = max_seq × count). At TOKEN_BUDGET=4096 the hidden
// state tensor is 4096×384×4B ≈ 6 MB regardless of batch count.
// For our workload (avg ~12 tokens/node), 4096 tokens ≈ 341 nodes — well under
// MAX_BATCH=512, so the budget is the effective limit in practice.
const MAX_BATCH: usize = 512;
const TOKEN_BUDGET: usize = 4_096;
// Commit to embed.db every TX_BATCH rows.
const TX_BATCH: usize = 5_000;

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

struct NomicPlugin {
    model: model::BgeModel,
    model_id: String,
    backend: String,
    /// HNSW index — None until first KNN call if not present at startup.
    index: Mutex<Option<index::VecIndex>>,
    index_path: PathBuf,
    /// graph.db path — for FTS candidate lookup in lazy embed path.
    db_path: PathBuf,
    /// embed.db path — for NOT-EXISTS filter + async persist of lazy embeds.
    embed_db_path: PathBuf,
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
        let dim = dim_for_model(model_id);
        let model = model::BgeModel::load(model_dir, dim).context("loading model")?;
        let index = index::VecIndex::try_load(&index_path).unwrap_or_else(|e| {
            tracing::warn!(
                "could not load HNSW index: {e:#} — KNN disabled until `travsr embed reindex` runs"
            );
            None
        });
        Ok(Self {
            model,
            model_id: model_id.to_owned(),
            backend: backend_label(model_id),
            index: Mutex::new(index),
            index_path,
            db_path,
            embed_db_path,
        })
    }
}

impl EmbedPlugin for NomicPlugin {
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn embedding_dim(&self) -> u32 {
        self.model.dim as u32
    }
    fn backend(&self) -> &str {
        &self.backend
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
    fn knn_impl(&self, req: &KnnRequest) -> Result<(Vec<i64>, Vec<f32>)> {
        let query_blob = self.model.embed_query(&req.query_text)?;
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
        let lazy_scored = self
            .lazy_embed_candidates(&req.query_text, &query_vec)
            .unwrap_or_else(|e| {
                tracing::debug!("lazy embed skipped (non-fatal): {e:#}");
                vec![]
            });

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
        let pairs: Vec<(i64, Vec<u8>)> =
            candidates.iter().map(|(nid, _)| *nid).zip(blobs).collect();
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

        let mut stmt = conn
            .prepare(
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
                 AND n.kind NOT IN \
                     ('file','file-module','import','module','field','variable') \
                 AND NOT EXISTS ( \
                     SELECT 1 FROM edb.node_embeddings e \
                     WHERE e.node_id = n.id AND e.model_id = ?2 \
                 ) \
                 ORDER BY rank \
                 LIMIT ?3",
            )
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
    pairs: &[(i64, Vec<u8>)],
    model_id: &str,
) -> Result<()> {
    let conn = Connection::open(embed_db_path).context("lazy persist: open embed.db")?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 30000;",
    )
    .context("lazy persist: configure embed.db")?;
    conn.execute("BEGIN", []).context("lazy persist: begin")?;
    let mut ins = conn
        .prepare(
            "INSERT OR IGNORE INTO node_embeddings (node_id, model_id, embedding) \
             VALUES (?1, ?2, ?3)",
        )
        .context("lazy persist: prepare insert")?;
    for (nid, blob) in pairs {
        ins.execute(rusqlite::params![nid, model_id, blob])
            .context("lazy persist: insert")?;
    }
    conn.execute("COMMIT", []).context("lazy persist: commit")?;
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

    let dim = dim_for_model(model_id);
    let model = if worker_mode {
        model::BgeModel::load_for_shard(model_dir, 1, dim).context("loading model (worker)")?
    } else {
        model::BgeModel::load(model_dir, dim).context("loading model")?
    };

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

    // Create schema in embed.db on first run (idempotent).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS edb.node_embeddings (
             node_id   INTEGER NOT NULL,
             model_id  TEXT    NOT NULL,
             embedding BLOB    NOT NULL,
             PRIMARY KEY (node_id, model_id)
         ) WITHOUT ROWID;
         CREATE INDEX IF NOT EXISTS edb.idx_node_embeddings_model
             ON node_embeddings(model_id);",
    )
    .context("create embed.db schema")?;

    // CDC: apply pending tombstones atomically — delete from embed.db, ack in graph.db.
    let tombstone_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM node_tombstones", [], |r| r.get(0))
        .unwrap_or(0);
    if tombstone_count > 0 {
        conn.execute_batch(
            "BEGIN;
             DELETE FROM edb.node_embeddings
                 WHERE node_id IN (SELECT node_id FROM node_tombstones);
             DELETE FROM node_tombstones;
             COMMIT;",
        )
        .context("applying CDC tombstones")?;
        tracing::info!(tombstone_count, "applied CDC tombstones to embed.db");
    }

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
    let kind_exclude = "'file','file-module','import','module','field','variable'";
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
         WHERE n.kind NOT IN ({kind_exclude}) \
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
        index::VecIndex::build_from_db(
            db_path,
            embed_db_path,
            model_id,
            &index_path,
            existing,
            dim_for_model(model_id),
        )
        .context("build_from_db")?;
        write_current_embed_model_meta(&conn, model_id)?;
        println!("Done — index saved to {}.", index_path.display());
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
                    dim_for_model(model_id),
                )
                .context("rebuild HNSW from existing embeddings before adding pending")?;
                index::VecIndex::try_load(&index_path)
                    .context("load freshly-rebuilt HNSW")?
                    .expect("just-rebuilt index must be loadable")
            } else {
                index::VecIndex::new_empty(&index_path, total, dim_for_model(model_id))
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
    let mut ins = conn.prepare(
        "INSERT OR REPLACE INTO edb.node_embeddings (node_id, model_id, embedding) \
         VALUES (?1, ?2, ?3)",
    )?;

    let mut tx_buffer: Vec<(i64, Vec<u8>)> = Vec::with_capacity(TX_BATCH + 512);
    let mut inserted = 0usize;

    let flush_buffer = |tx_buffer: &Vec<(i64, Vec<u8>)>,
                        conn: &rusqlite::Connection,
                        ins: &mut rusqlite::Statement<'_>,
                        idx: &Option<index::VecIndex>|
     -> Result<()> {
        conn.execute("BEGIN", [])?;
        for (node_id, blob) in tx_buffer {
            ins.execute(rusqlite::params![node_id, model_id, blob])?;
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

        for ((node_id, _), blob) in chunk.iter().zip(blobs.iter()) {
            tx_buffer.push((*node_id, blob.clone()));
        }

        if tx_buffer.len() >= TX_BATCH {
            flush_buffer(&tx_buffer, &conn, &mut ins, &idx)?;
            inserted += tx_buffer.len();
            if inserted % 1_000 < tx_buffer.len() || inserted >= total {
                println!("  embedded {inserted}/{total}");
            }
            tx_buffer.clear();
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
fn reindex_parallel(
    model_dir: &Path,
    db_path: &Path,
    embed_db_path: &Path,
    parallel: usize,
    busy_timeout_ms: u64,
    phase: Phase,
    model_id: &str,
) -> Result<()> {
    tracing::info!(
        parallel,
        db = %db_path.display(),
        embed_db = %embed_db_path.display(),
        "parallel reindex: {} reader thread(s), single model load",
        parallel
    );

    // ── Step 1: CDC tombstones (main thread, needs write access to both dbs) ──
    {
        let conn = Connection::open(db_path).context("open graph.db for tombstones")?;
        let embed_str = embed_db_path
            .to_str()
            .context("embed.db path is not valid UTF-8")?;
        let escaped = embed_str.replace('\'', "''");
        conn.execute_batch(&format!(
            "PRAGMA busy_timeout = {busy_timeout_ms};
             ATTACH DATABASE '{escaped}' AS edb;
             PRAGMA edb.journal_mode = WAL;"
        ))
        .context("configure connections for tombstone pass")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS edb.node_embeddings (
                 node_id   INTEGER NOT NULL,
                 model_id  TEXT    NOT NULL,
                 embedding BLOB    NOT NULL,
                 PRIMARY KEY (node_id, model_id)
             ) WITHOUT ROWID;
             CREATE INDEX IF NOT EXISTS edb.idx_node_embeddings_model
                 ON node_embeddings(model_id);",
        )
        .context("create embed.db schema")?;
        let tombstone_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM node_tombstones", [], |r| r.get(0))
            .unwrap_or(0);
        if tombstone_count > 0 {
            conn.execute_batch(
                "BEGIN;
                 DELETE FROM edb.node_embeddings
                     WHERE node_id IN (SELECT node_id FROM node_tombstones);
                 DELETE FROM node_tombstones;
                 COMMIT;",
            )
            .context("applying CDC tombstones")?;
            tracing::info!(tombstone_count, "applied CDC tombstones to embed.db");
        }
    }

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
        let kind_exclude = "'file','file-module','import','module','field','variable'";
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
             WHERE n.kind NOT IN ({kind_exclude}) \
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

    let n_workers = parallel.min(n_batches).max(1);
    let model =
        model::BgeModel::load(model_dir, dim_for_model(model_id)).context("loading model")?;
    tracing::info!(
        total,
        n_batches,
        n_workers,
        "model loaded; spawning {} inference threads (shared batch queue)",
        n_workers
    );

    // ── Steps 4-6: N consumer threads ────────────────────────────────────────
    // Workers share Arc<Vec<items>>, Arc<Vec<ranges>>, Arc<AtomicUsize>.
    // BgeModel::clone() is cheap (Arc<TypedRunnableModel> + Tokenizer clone).
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
                    wconn
                        .execute_batch(
                            "CREATE TABLE IF NOT EXISTS node_embeddings (
                                 node_id   INTEGER NOT NULL,
                                 model_id  TEXT    NOT NULL,
                                 embedding BLOB    NOT NULL,
                                 PRIMARY KEY (node_id, model_id)
                             ) WITHOUT ROWID;
                             CREATE INDEX IF NOT EXISTS idx_node_embeddings_model
                                 ON node_embeddings(model_id);",
                        )
                        .context("worker: ensure schema")?;
                    let mut ins = wconn
                        .prepare(
                            "INSERT OR REPLACE INTO node_embeddings \
                             (node_id, model_id, embedding) VALUES (?1, ?2, ?3)",
                        )
                        .context("worker: prepare insert")?;

                    let mut tx_buf: Vec<(i64, Vec<u8>)> = Vec::with_capacity(TX_BATCH + 512);
                    let mut inserted = 0usize;

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
                        for ((nid, _), blob) in batch.iter().zip(blobs.iter()) {
                            tx_buf.push((*nid, blob.clone()));
                        }
                        if tx_buf.len() >= TX_BATCH {
                            wconn.execute("BEGIN", []).context("worker: begin")?;
                            for (nid, blob) in &tx_buf {
                                ins.execute(rusqlite::params![nid, mid_w.as_str(), blob])
                                    .context("worker: insert")?;
                            }
                            wconn.execute("COMMIT", []).context("worker: commit")?;
                            inserted += tx_buf.len();
                            tx_buf.clear();
                        }
                    }

                    if !tx_buf.is_empty() {
                        wconn.execute("BEGIN", []).context("worker: begin final")?;
                        for (nid, blob) in &tx_buf {
                            ins.execute(rusqlite::params![nid, mid_w.as_str(), blob])
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
    let conn = Connection::open(db_path).context("open graph.db")?;
    let embed_db_str = embed_db_path
        .to_str()
        .context("embed.db path is not valid UTF-8")?;
    conn.execute_batch(&format!("ATTACH DATABASE '{embed_db_str}' AS edb"))
        .context("attach embed.db")?;

    let existing: usize = conn
        .query_row(
            "SELECT COUNT(*) FROM edb.node_embeddings e \
             JOIN nodes n ON n.id = e.node_id \
             WHERE e.model_id = ?1 \
             AND n.kind NOT IN ('file', 'file-module', 'import', 'module', 'field', 'variable')",
            [model_id],
            |r| r.get(0),
        )
        .context("counting existing meaningful embeddings")?;
    anyhow::ensure!(
        existing > 0,
        "no embeddings in embed.db — run `travsr embed reindex` first"
    );
    let index_path = index_path_for_db(db_path, model_id);
    println!("Building HNSW index from {existing} embeddings...");
    index::VecIndex::build_from_db(
        db_path,
        embed_db_path,
        model_id,
        &index_path,
        existing,
        dim_for_model(model_id),
    )
    .context("build_from_db")?;
    write_current_embed_model_meta(&conn, model_id)?;
    println!("Done — index saved to {}.", index_path.display());
    tracing::info!(existing, "HNSW index rebuilt");
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
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!(
                    "usage: travsr-embed-nomic \
                     [--model-id <id>] \
                     [--reindex <db> [--embed-db <embed.db>] \
                      [--row-start <i64> --row-end <i64>] \
                      [--phase1 <n>|--phase2 <n>] [--shard <i>/<n>] \
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
