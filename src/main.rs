// travsr-embed-nomic — RFC-018 embedding sidecar.
//
// Modes controlled by argv:
//
//   (no args / --db-path)      Daemon mode: speak EmbedPlugin IPC over stdio.
//                              Requires --db-path <graph.db> so the sidecar
//                              knows which repo's per-repo HNSW index to load.
//
//   --reindex <db>             One-shot: embed all pending nodes in graph.db,
//                              write node_embeddings rows, and update the
//                              per-repo HNSW index.  Exits when done.
//
//   --reindex <db> --shard <i>/<n>
//                              One-shot shard: embed only nodes where id % n = i.
//                              Skips HNSW writes — the CLI orchestrator calls
//                              --rebuild-index when all shards complete.
//
//   --rebuild-index <db>       Rebuild per-repo HNSW from node_embeddings.
//                              No ONNX inference — pure SQLite stream.
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
// Commit to SQLite every TX_BATCH rows: limits WAL growth during long runs.
const TX_BATCH: usize = 5_000;

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
fn estimate_tokens(kind: &str, sig: &str) -> usize {
    (kind.len() + sig.len() + 2) / 4 + 4
}

// ── --reindex mode ────────────────────────────────────────────────────────────

/// One-shot embedding: read all nodes from graph.db that do not yet have a
/// nomic-v1.5-int8 row in node_embeddings, embed them in token-budget-bounded
/// batches, write the BLOBs, and update the per-repo HNSW index.
///
/// When `shard = Some((i, n))`, only processes nodes where `id % n = i` and
/// skips all HNSW operations — the CLI orchestrator calls rebuild_index() when
/// all n shards have finished.
fn reindex(model_dir: &Path, db_path: &Path, shard: Option<(usize, usize)>) -> Result<()> {
    let shard_mode = shard.is_some();
    let (shard_idx, n_shards) = shard.unwrap_or((0, 1));

    tracing::info!(
        db = %db_path.display(),
        shard_idx,
        n_shards,
        "starting reindex"
    );

    // In shard mode, divide threads across shard processes to avoid oversubscription.
    // N shards × (cores/N) threads = cores total — same utilisation, no thrashing.
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let model = if shard_mode {
        let t = (cores / n_shards).max(1);
        tracing::debug!(intra_threads = t, "shard mode: limiting ORT threads");
        model::NomicModel::load_for_shard(model_dir, t).context("loading model (shard)")?
    } else {
        model::NomicModel::load(model_dir).context("loading model")?
    };
    let conn = Connection::open(db_path).context("open graph.db")?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA cache_size = -16384;
         PRAGMA busy_timeout = 120000;",
    )
    .context("configure SQLite pragmas")?;

    // Use ((id % n) + n) % n for unsigned-safe modulo: SQLite's % returns negative
    // values for negative inputs, so bare `id % n` misses nodes with negative IDs
    // (Kythe VName hashes are i64 and can be negative).
    let mut stmt = conn.prepare(
        "SELECT n.id, n.kind, n.signature \
         FROM nodes n \
         WHERE NOT EXISTS ( \
             SELECT 1 FROM node_embeddings e \
             WHERE e.node_id = n.id AND e.model_id = ?1 \
         ) AND (((n.id % ?2) + ?2) % ?2 = ?3)",
    )?;
    let pending: Vec<(i64, String, String)> = stmt
        .query_map(
            rusqlite::params![MODEL_ID, n_shards as i64, shard_idx as i64],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?
        .filter_map(|r| r.ok())
        .collect();

    let total = pending.len();
    tracing::info!(total, shard_idx, n_shards, "nodes to embed");

    let index_path = index_path_for_db(db_path);

    if total == 0 {
        if shard_mode {
            println!("  shard {shard_idx}/{n_shards}: no pending nodes.");
            return Ok(());
        }
        // Non-shard: if the index file is also present, nothing to do.
        // If missing (e.g. accidental deletion), rebuild from node_embeddings.
        if index_path.exists() {
            write_current_embed_model_meta(&conn)?;
            println!("All nodes already have embeddings for {MODEL_ID}. Index up to date.");
            return Ok(());
        }
        let existing: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM node_embeddings WHERE model_id = ?1",
                [MODEL_ID],
                |r| r.get(0),
            )
            .unwrap_or(0);
        println!("All nodes already embedded ({existing} rows). Building missing HNSW index...");
        index::VecIndex::build_from_db(db_path, MODEL_ID, &index_path, existing)
            .context("build_from_db")?;
        write_current_embed_model_meta(&conn)?;
        println!("Done — index saved to {}.", index_path.display());
        return Ok(());
    }

    // In shard mode, skip HNSW entirely — orchestrator rebuilds after all shards.
    let idx: Option<index::VecIndex> = if shard_mode {
        None
    } else {
        let idx = if index_path.exists() {
            index::VecIndex::try_load(&index_path)
                .context("load existing HNSW index")?
                .expect("index file exists but load returned None")
        } else {
            // Index file missing but DB may already have embeddings (e.g. accidental
            // deletion after a partial reindex): rebuild first so existing nodes are
            // not silently dropped from KNN, then the pending loop appends new ones.
            let existing: usize = conn
                .query_row(
                    "SELECT COUNT(*) FROM node_embeddings WHERE model_id = ?1",
                    [MODEL_ID],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            if existing > 0 {
                index::VecIndex::build_from_db(db_path, MODEL_ID, &index_path, existing)
                    .context("rebuild HNSW from existing embeddings before adding pending")?;
                index::VecIndex::try_load(&index_path)
                    .context("load freshly-rebuilt HNSW")?
                    .expect("just-rebuilt index must be loadable")
            } else {
                index::VecIndex::new_empty(&index_path, total)
                    .context("create new HNSW index")?
            }
        };
        // load() freezes capacity at save time; re-reserve before inserting.
        idx.reserve(idx.size() + total)
            .context("reserve HNSW capacity for pending nodes")?;
        Some(idx)
    };

    // ── bin-pack pending nodes into token-budget-bounded batches ──────────────
    //
    // Cost model: BatchLongest pads all items to max_seq × count tokens.
    // estimate_tokens() uses byte_len / 4 as a cheap proxy — accurate enough.
    let est_lens: Vec<usize> = pending
        .iter()
        .map(|(_, kind, sig)| estimate_tokens(kind, sig))
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
        if batch_start < pending.len() {
            batch_ranges.push(batch_start..pending.len());
        }
    }

    // Prepare INSERT once; reuse across all transactions.
    let mut ins = conn.prepare(
        "INSERT OR REPLACE INTO node_embeddings (node_id, model_id, embedding) \
         VALUES (?1, ?2, ?3)",
    )?;

    // Accumulate (node_id, blob) pairs from ONNX inference runs, then flush to
    // SQLite in a short write transaction.  ONNX inference runs with NO open
    // transaction so the WAL write lock is not held during the expensive GPU/CPU
    // inference step — this lets parallel shard processes interleave writes.
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
        let chunk = &pending[range];
        let texts: Vec<String> = chunk
            .iter()
            .map(|(_, kind, sig)| format!("{kind}: {sig}"))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        // ONNX inference: no transaction open, write lock not held.
        let blobs = model.embed_documents(&text_refs)?;

        for ((node_id, _, _), blob) in chunk.iter().zip(blobs.iter()) {
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

    // Flush remaining items.
    if !tx_buffer.is_empty() {
        flush_buffer(&tx_buffer, &conn, &mut ins, &idx)?;
        inserted += tx_buffer.len();
        println!("  embedded {inserted}/{total}");
        tx_buffer.clear();
    }

    if let Some(ref idx_inner) = idx {
        idx_inner.save()?;
        write_current_embed_model_meta(&conn)?;
        println!(
            "Done — {inserted} nodes embedded. Index saved to {}.",
            index_path.display()
        );
    } else {
        println!("  shard {shard_idx}/{n_shards}: {inserted} nodes embedded.");
    }

    tracing::info!(inserted, total, shard_idx, n_shards, "reindex complete");
    Ok(())
}

// ── --rebuild-index mode ──────────────────────────────────────────────────────

/// Rebuild the per-repo HNSW index by streaming all rows from node_embeddings.
/// No ONNX inference — pure SQLite I/O.  Used as the final step by the CLI
/// orchestrator after parallel shard embedding completes.
fn rebuild_index(db_path: &Path) -> Result<()> {
    tracing::info!(db = %db_path.display(), "rebuilding HNSW index");
    let conn = Connection::open(db_path).context("open graph.db")?;
    let existing: usize = conn
        .query_row(
            "SELECT COUNT(*) FROM node_embeddings WHERE model_id = ?1",
            [MODEL_ID],
            |r| r.get(0),
        )
        .context("counting existing embeddings")?;
    anyhow::ensure!(
        existing > 0,
        "no embeddings in node_embeddings — run `travsr embed reindex` first"
    );
    let index_path = index_path_for_db(db_path);
    println!("Building HNSW index from {existing} embeddings...");
    index::VecIndex::build_from_db(db_path, MODEL_ID, &index_path, existing)
        .context("build_from_db")?;
    write_current_embed_model_meta(&conn)?;
    println!("Done — index saved to {}.", index_path.display());
    tracing::info!(existing, "HNSW index rebuilt");
    Ok(())
}

// ── entry point ───────────────────────────────────────────────────────────────

fn main() {
    // Structured logging to stderr — the daemon reads stdout for IPC.
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

    // Parse argv into a flag map: --flag value pairs.
    let args: Vec<String> = std::env::args().collect();
    let mut reindex_db: Option<PathBuf> = None;
    let mut daemon_db: Option<PathBuf> = None;
    let mut rebuild_db: Option<PathBuf> = None;
    let mut shard: Option<(usize, usize)> = None;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--reindex" => {
                i += 1;
                reindex_db = Some(args.get(i).map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("usage: travsr-embed-nomic --reindex <graph.db-path>");
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
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!(
                    "usage: travsr-embed-nomic \
                     [--reindex <db> [--shard <i>/<n>]] \
                     [--rebuild-index <db>] \
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
    if rebuild_db.is_some() && reindex_db.is_some() {
        eprintln!("--rebuild-index and --reindex are mutually exclusive");
        std::process::exit(1);
    }

    if let Some(db_path) = rebuild_db {
        if let Err(e) = rebuild_index(&db_path) {
            eprintln!("rebuild-index failed: {e:#}");
            std::process::exit(1);
        }
    } else if let Some(db_path) = reindex_db {
        if let Err(e) = reindex(&model_dir, &db_path, shard) {
            eprintln!("reindex failed: {e:#}");
            std::process::exit(1);
        }
    } else {
        // Daemon / IPC mode.  --db-path is required so we know which per-repo
        // HNSW to load.
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
