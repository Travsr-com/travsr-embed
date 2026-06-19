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
//                        node_embeddings, and exit.  Used to bootstrap the
//                        embedding table before the daemon builds its HNSW index.
//                        Run: ~/.travsr/bin/travsr-embed-nomic \
//                               --reindex ~/.travsr/<repo>/graph.db

#![forbid(unsafe_code)]

mod index;
mod model;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context as _, Result};
use travsr_plugin_protocol::{EmbedPlugin, EmbedRequest, EmbedResponse, KnnRequest, KnnResponse};
use travsr_plugin_sdk::run_embed_plugin;

const MODEL_ID: &str = "nomic-v1.5-int8";
const BACKEND: &str = "nomic-embed-text-v1.5 int8 MRL-256";
const EMBED_DIM: u32 = 256;
const MAX_BATCH: u32 = 32; // conservative for int8 model on CPU

// ── Plugin struct ─────────────────────────────────────────────────────────────

struct NomicPlugin {
    model: model::NomicModel,
    /// Lazily-built HNSW index (None until first KNN request).
    index: Mutex<Option<index::VecIndex>>,
}

impl NomicPlugin {
    fn load(model_dir: &Path) -> Result<Self> {
        let model = model::NomicModel::load(model_dir).context("loading model")?;
        Ok(Self {
            model,
            index: Mutex::new(None),
        })
    }
}

impl EmbedPlugin for NomicPlugin {
    fn model_id(&self) -> &str { MODEL_ID }
    fn embedding_dim(&self) -> u32 { EMBED_DIM }
    fn backend(&self) -> &str { BACKEND }
    fn max_batch(&self) -> u32 { MAX_BATCH }

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
            Ok((ids, scores)) => KnnResponse { node_ids: ids, scores },
            Err(e) => {
                tracing::warn!("knn failed (non-fatal): {e:#}");
                KnnResponse { node_ids: vec![], scores: vec![] }
            }
        }
    }
}

impl NomicPlugin {
    fn knn_impl(&self, req: &KnnRequest) -> Result<(Vec<i64>, Vec<f32>)> {
        // Embed the query with the search_query prefix.
        let query_blob = self.model.embed_query(&req.query_text)?;

        // Build the HNSW index on first call, then reuse it.
        let mut guard = self.index.lock().map_err(|_| anyhow::anyhow!("index mutex poisoned"))?;
        if guard.is_none() {
            tracing::info!(db = %req.db_path.display(), "building HNSW index on first KNN call");
            *guard = Some(
                index::VecIndex::build(&req.db_path, &req.model_id)
                    .context("building vec index")?,
            );
        }
        let idx = guard.as_mut().unwrap();

        if idx.count() == 0 {
            tracing::debug!("index is empty — no embeddings indexed yet");
            return Ok((vec![], vec![]));
        }

        let raw = idx.knn(&query_blob, req.k)?;
        let ids: Vec<i64> = raw.iter().map(|&(id, _)| id).collect();
        // sqlite-vec returns L2 / cosine distance; convert distance to similarity score.
        // For cosine distance d: similarity = 1 - d  (stored vecs are L2-normalised).
        let scores: Vec<f32> = raw.iter().map(|&(_, d)| (1.0 - d).clamp(0.0, 1.0)).collect();
        Ok((ids, scores))
    }
}

// ── --reindex mode ────────────────────────────────────────────────────────────

/// One-shot embedding: read all nodes from graph.db that do not yet have a
/// nomic-v1.5-int8 row in node_embeddings, embed them in batches, and write
/// the results back.  Exits when done; does not start the IPC loop.
///
/// Embedding text per node: "{kind}: {signature}"
/// This is the most semantic content available without a stored snippet.
fn reindex(model_dir: &Path, db_path: &Path) -> Result<()> {
    use rusqlite::Connection;

    tracing::info!(db = %db_path.display(), "starting reindex");

    let model = model::NomicModel::load(model_dir).context("loading model")?;
    let conn = Connection::open(db_path).context("open graph.db")?;

    // Find nodes without an embedding for this model.
    let mut stmt = conn.prepare(
        "SELECT n.id, n.kind, n.signature \
         FROM nodes n \
         WHERE NOT EXISTS ( \
             SELECT 1 FROM node_embeddings e \
             WHERE e.node_id = n.id AND e.model_id = ?1 \
         )",
    )?;

    let pending: Vec<(i64, String, String)> = stmt
        .query_map([MODEL_ID], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .filter_map(|r| r.ok())
        .collect();

    let total = pending.len();
    tracing::info!(total, "nodes to embed");

    if total == 0 {
        println!("All nodes already have embeddings for {MODEL_ID}.");
        return Ok(());
    }

    let mut inserted = 0usize;
    for chunk in pending.chunks(MAX_BATCH as usize) {
        let texts: Vec<String> = chunk
            .iter()
            .map(|(_, kind, sig)| format!("{kind}: {sig}"))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let blobs = model.embed_documents(&text_refs)?;

        conn.execute("BEGIN", [])?;
        {
            let mut ins = conn.prepare(
                "INSERT OR REPLACE INTO node_embeddings (node_id, model_id, embedding) \
                 VALUES (?1, ?2, ?3)",
            )?;
            for ((node_id, _, _), blob) in chunk.iter().zip(blobs.iter()) {
                ins.execute(rusqlite::params![node_id, MODEL_ID, blob])?;
            }
        }
        conn.execute("COMMIT", [])?;

        inserted += chunk.len();
        if inserted % 1000 == 0 || inserted == total {
            println!("  embedded {inserted}/{total}");
        }
    }

    println!("Done — {inserted} nodes embedded with model '{MODEL_ID}'.");
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
        None => {
            // Daemon IPC mode.
            match NomicPlugin::load(&model_dir) {
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
            }
        }
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
