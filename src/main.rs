// travsr-embed-nomic — RFC-018 embedding sidecar.
//
// Two modes controlled by argv:
//
//   (no args)            Daemon mode: speak EmbedPlugin IPC over stdio.
//                        The travsr daemon spawns this binary and communicates
//                        over framed-JSON (EmbedPluginRequest / Response).
//
//   --reindex <db-path>  One-shot mode: read every node from graph.db, embed
//                        all that don't already have a nomic-v1.5-int8 row in
//                        node_embeddings, write them back, and update the
//                        hnsw.usearch index file.  Exits when done.
//                        Run: ~/.travsr/bin/travsr-embed-nomic \
//                               --reindex ~/.travsr/<repo>/graph.db

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
// int8 ONNX at batch=64, worst-case seq=512: 64×512×768×4B = 134 MB — safe on any machine.
const MAX_BATCH: u32 = 64;
// Commit to SQLite every TX_BATCH rows: 2 fsyncs per 10k nodes vs. 313 with per-chunk commits.
const TX_BATCH: usize = 5_000;

// ── Plugin struct ─────────────────────────────────────────────────────────────

struct NomicPlugin {
    model: model::NomicModel,
    /// HNSW index — None until first KNN call if not present at startup.
    index: Mutex<Option<index::VecIndex>>,
    index_path: PathBuf,
}

impl NomicPlugin {
    fn load(model_dir: &Path) -> Result<Self> {
        let model = model::NomicModel::load(model_dir).context("loading model")?;
        let index_path = model_dir.join("hnsw.usearch");
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
        MAX_BATCH
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

        // mtime check + search are both inside VecIndex::knn().
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

// ── --reindex mode ────────────────────────────────────────────────────────────

fn write_current_embed_model_meta(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('current_embed_model', ?1)",
        [MODEL_ID],
    )
    .context("writing current_embed_model meta")?;
    Ok(())
}

/// One-shot embedding: read all nodes from graph.db that do not yet have a
/// nomic-v1.5-int8 row in node_embeddings, embed them in batches, write the
/// BLOBs, and update hnsw.usearch with the new vectors.  Exits when done.
fn reindex(model_dir: &Path, db_path: &Path) -> Result<()> {
    tracing::info!(db = %db_path.display(), "starting reindex");

    let model = model::NomicModel::load(model_dir).context("loading model")?;
    let conn = Connection::open(db_path).context("open graph.db")?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA cache_size = -16384;",
    )
    .context("configure SQLite pragmas")?;

    // Collect only the nodes that still need embedding.
    let mut stmt = conn.prepare(
        "SELECT n.id, n.kind, n.signature \
         FROM nodes n \
         WHERE NOT EXISTS ( \
             SELECT 1 FROM node_embeddings e \
             WHERE e.node_id = n.id AND e.model_id = ?1 \
         )",
    )?;
    let pending: Vec<(i64, String, String)> = stmt
        .query_map([MODEL_ID], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let total = pending.len();
    tracing::info!(total, "nodes to embed");

    let index_path = model_dir.join("hnsw.usearch");

    if total == 0 {
        // All nodes already embedded. If the index file is also present, nothing to do.
        // If the index file is missing (e.g. first run after upgrading from hnsw_rs, or
        // after accidental deletion), rebuild it by streaming from node_embeddings.
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

    // Open or create the usearch HNSW index.
    // When the index file is missing but the DB already has embeddings (e.g. accidental
    // deletion after a partial reindex), rebuild from the existing rows first so those
    // nodes are not silently dropped from KNN. The pending loop then appends new ones.
    let idx = if index_path.exists() {
        index::VecIndex::try_load(&index_path)
            .context("load existing HNSW index")?
            .expect("index file exists but load returned None")
    } else {
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
            index::VecIndex::new_empty(&index_path, total).context("create new HNSW index")?
        }
    };

    // Ensure the index has room for all pending nodes on top of whatever is
    // already there.  load() freezes capacity at save time, so we must
    // re-reserve before inserting.  new_empty() and build_from_db() already
    // reserve enough for their own count, but not for the pending additions.
    idx.reserve(idx.size() + total)
        .context("reserve HNSW capacity for pending nodes")?;

    // Prepare INSERT once; reuse across all chunks.
    let mut ins = conn.prepare(
        "INSERT OR REPLACE INTO node_embeddings (node_id, model_id, embedding) \
         VALUES (?1, ?2, ?3)",
    )?;

    let mut inserted = 0usize;
    let mut tx_rows = 0usize;
    conn.execute("BEGIN", [])?;

    for chunk in pending.chunks(MAX_BATCH as usize) {
        let texts: Vec<String> = chunk
            .iter()
            .map(|(_, kind, sig)| format!("{kind}: {sig}"))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let blobs = model.embed_documents(&text_refs)?;

        for ((node_id, _, _), blob) in chunk.iter().zip(blobs.iter()) {
            ins.execute(rusqlite::params![node_id, MODEL_ID, blob])?;
            let vec = model::blob_to_f32(blob);
            idx.add(*node_id, &vec)?;
            tx_rows += 1;
        }

        if tx_rows >= TX_BATCH {
            conn.execute("COMMIT", [])?;
            conn.execute("BEGIN", [])?;
            tx_rows = 0;
        }

        inserted += blobs.len().min(chunk.len());
        if inserted % 1_000 == 0 || inserted == total {
            println!("  embedded {inserted}/{total}");
        }
    }

    conn.execute("COMMIT", [])?;
    idx.save()?;
    write_current_embed_model_meta(&conn)?;

    println!(
        "Done — {inserted} nodes embedded. Index saved to {}.",
        index_path.display()
    );
    tracing::info!(inserted, total, "reindex complete");
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

    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--reindex") => {
            let db_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
                eprintln!("usage: travsr-embed-nomic --reindex <graph.db-path>");
                std::process::exit(1);
            });
            if let Err(e) = reindex(&model_dir, &db_path) {
                eprintln!("reindex failed: {e:#}");
                std::process::exit(1);
            }
        }
        None => match NomicPlugin::load(&model_dir) {
            Ok(plugin) => {
                tracing::info!(
                    model_dir = %model_dir.display(),
                    model_id  = MODEL_ID,
                    "embed sidecar ready"
                );
                run_embed_plugin(plugin);
            }
            Err(e) => {
                eprintln!("travsr-embed: startup failed: {e:#}");
                std::process::exit(1);
            }
        },
        Some(other) => {
            eprintln!("unknown argument: {other}");
            eprintln!("usage: travsr-embed-nomic [--reindex <db-path>]");
            std::process::exit(1);
        }
    }
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
