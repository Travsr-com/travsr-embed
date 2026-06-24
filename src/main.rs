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

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context as _, Result};
use rusqlite::Connection;
use travsr_plugin_protocol::{EmbedPlugin, EmbedRequest, EmbedResponse, KnnRequest, KnnResponse};
use travsr_plugin_sdk::run_embed_plugin;

const MODEL_ID: &str = "nomic-v1.5-int8";
const BACKEND: &str = "nomic-embed-text-v1.5 int8 MRL-256";
const EMBED_DIM: u32 = 256;
// MAX_BATCH: hard cap on items per ONNX forward pass.
// TOKEN_BUDGET: soft cap on padded tensor cost (BatchLongest pads all items to
// max_seq_in_batch; cost = max_seq × count). At TOKEN_BUDGET=4096 the hidden
// state tensor is 4096×768×4B ≈ 12 MB regardless of batch count.
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
    model: model::NomicModel,
    /// HNSW index — None until first KNN call if not present at startup.
    index: Mutex<Option<index::VecIndex>>,
    index_path: PathBuf,
}

impl NomicPlugin {
    /// `model_dir`  — global model directory (ONNX + tokenizer files)
    /// `index_path` — per-repo HNSW file (derived from db_path by the caller)
    fn load(model_dir: &Path, index_path: PathBuf) -> Result<Self> {
        let model = model::NomicModel::load(model_dir).context("loading model")?;
        let index = index::VecIndex::try_load(&index_path).unwrap_or_else(|e| {
            tracing::warn!(
                "could not load HNSW index: {e:#} — KNN disabled until `travsr embed reindex` runs"
            );
            None
        });
        Ok(Self {
            model,
            index: Mutex::new(index),
            index_path,
        })
    }
}

impl EmbedPlugin for NomicPlugin {
    fn model_id(&self) -> &str {
        MODEL_ID
    }
    fn embedding_dim(&self) -> u32 {
        EMBED_DIM
    }
    fn backend(&self) -> &str {
        BACKEND
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

        let mut guard = self
            .index
            .lock()
            .map_err(|_| anyhow::anyhow!("index mutex poisoned"))?;

        // Late-load: if the daemon started before reindex ran, pick up the index now.
        if guard.is_none() && self.index_path.exists() {
            *guard = index::VecIndex::try_load(&self.index_path)?;
        }

        let idx = match guard.as_mut() {
            None => {
                tracing::debug!("no HNSW index — run `travsr embed reindex`");
                return Ok((vec![], vec![]));
            }
            Some(i) => i,
        };

        let raw = idx.knn(&query_blob, req.k)?;
        let ids: Vec<i64> = raw.iter().map(|&(id, _)| id).collect();
        // usearch returns cosine distance; convert to similarity score.
        let scores: Vec<f32> = raw
            .iter()
            .map(|&(_, d)| (1.0 - d).clamp(0.0, 1.0))
            .collect();
        Ok((ids, scores))
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn write_current_embed_model_meta(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('current_embed_model', ?1)",
        [MODEL_ID],
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
fn reindex(
    model_dir: &Path,
    db_path: &Path,
    embed_db_path: &Path,
    shard: Option<(usize, usize)>,
    row_range: Option<(i64, i64)>,
    busy_timeout_ms: u64,
    phase: Phase,
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

    let model = if worker_mode {
        model::NomicModel::load_for_shard(model_dir, 1).context("loading model (worker)")?
    } else {
        model::NomicModel::load(model_dir).context("loading model")?
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
    let sql = format!(
        "SELECT n.id, n.kind, n.signature \
         FROM nodes n \
         WHERE n.kind NOT IN ('file', 'file-module', 'import', 'module', 'field', 'variable') \
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
        .query_map([MODEL_ID], |row| {
            let id: i64 = row.get(0)?;
            let kind: String = row.get(1)?;
            let sig: String = row.get(2)?;
            Ok((id, format!("{kind}: {sig}")))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let total = pending.len();
    tracing::info!(total, worker = %worker_label, "symbol nodes to embed");

    let mut texts = pending;
    texts.sort_by_key(|(_, text)| estimate_tokens(text));

    let index_path = index_path_for_db(db_path);

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
            write_current_embed_model_meta(&conn)?;
            println!("All nodes already have embeddings for {MODEL_ID}. Index up to date.");
            return Ok(());
        }
        let existing: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM edb.node_embeddings WHERE model_id = ?1",
                [MODEL_ID],
                |r| r.get(0),
            )
            .unwrap_or(0);
        println!("All nodes already embedded ({existing} rows). Building missing HNSW index...");
        index::VecIndex::build_from_db(db_path, embed_db_path, MODEL_ID, &index_path, existing)
            .context("build_from_db")?;
        write_current_embed_model_meta(&conn)?;
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
                    [MODEL_ID],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            if existing > 0 {
                index::VecIndex::build_from_db(db_path, embed_db_path, MODEL_ID, &index_path, existing)
                    .context("rebuild HNSW from existing embeddings before adding pending")?;
                index::VecIndex::try_load(&index_path)
                    .context("load freshly-rebuilt HNSW")?
                    .expect("just-rebuilt index must be loadable")
            } else {
                index::VecIndex::new_empty(&index_path, total).context("create new HNSW index")?
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
            ins.execute(rusqlite::params![node_id, MODEL_ID, blob])?;
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
        rebuild_index(db_path, embed_db_path)?;
    }

    tracing::info!(inserted, total, worker = %worker_label, "reindex complete");
    Ok(())
}

// ── --rebuild-index mode ──────────────────────────────────────────────────────

/// Rebuild the per-repo HNSW index by streaming all rows from embed.db.node_embeddings.
/// No ONNX inference — pure SQLite I/O.  Used as the final step by the CLI
/// orchestrator after parallel shard embedding completes.
fn rebuild_index(db_path: &Path, embed_db_path: &Path) -> Result<()> {
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
            [MODEL_ID],
            |r| r.get(0),
        )
        .context("counting existing meaningful embeddings")?;
    anyhow::ensure!(
        existing > 0,
        "no embeddings in embed.db — run `travsr embed reindex` first"
    );
    let index_path = index_path_for_db(db_path);
    println!("Building HNSW index from {existing} embeddings...");
    index::VecIndex::build_from_db(db_path, embed_db_path, MODEL_ID, &index_path, existing)
        .context("build_from_db")?;
    write_current_embed_model_meta(&conn)?;
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

    let model_dir = match model_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("travsr-embed: cannot find model dir: {e:#}");
            eprintln!("  Run: travsr embed init");
            std::process::exit(1);
        }
    };

    let args: Vec<String> = std::env::args().collect();
    let mut reindex_db: Option<PathBuf> = None;
    let mut daemon_db: Option<PathBuf> = None;
    let mut rebuild_db: Option<PathBuf> = None;
    let mut embed_db: Option<PathBuf> = None;
    let mut shard: Option<(usize, usize)> = None;
    let mut row_start: Option<i64> = None;
    let mut row_end: Option<i64> = None;
    let mut busy_timeout_ms: u64 = 120_000;
    let mut phase = Phase::All;
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
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!(
                    "usage: travsr-embed-nomic \
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
    if rebuild_db.is_some() && reindex_db.is_some() {
        eprintln!("--rebuild-index and --reindex are mutually exclusive");
        std::process::exit(1);
    }

    if let Some(db_path) = rebuild_db {
        let embed_path = embed_db.unwrap_or_else(|| embed_db_path_for(&db_path));
        if let Err(e) = rebuild_index(&db_path, &embed_path) {
            eprintln!("rebuild-index failed: {e:#}");
            std::process::exit(1);
        }
    } else if let Some(db_path) = reindex_db {
        let embed_path = embed_db.unwrap_or_else(|| embed_db_path_for(&db_path));
        let row_range = row_start.zip(row_end);
        if let Err(e) = reindex(&model_dir, &db_path, &embed_path, shard, row_range, busy_timeout_ms, phase) {
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
        let index_path = index_path_for_db(&db_path);
        match NomicPlugin::load(&model_dir, index_path) {
            Ok(plugin) => {
                tracing::info!(
                    model_dir = %model_dir.display(),
                    model_id  = MODEL_ID,
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

/// Per-repo HNSW index path, co-located with graph.db.
/// Node IDs are SQLite rowids scoped to one db, so each repo needs its own index.
fn index_path_for_db(db_path: &Path) -> PathBuf {
    let dir = db_path.parent().unwrap_or(db_path);
    dir.join(format!("{MODEL_ID}.hnsw.usearch"))
}

fn model_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("HOME not set"))?;
    let dir = home.join(".travsr").join("models").join(MODEL_ID);
    anyhow::ensure!(
        dir.exists(),
        "model directory not found: {}\n  Run: travsr embed init",
        dir.display()
    );
    Ok(dir)
}
