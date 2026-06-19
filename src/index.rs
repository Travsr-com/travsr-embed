// In-memory HNSW index for O(log n) approximate nearest-neighbour search over
// node embeddings, backed by hnsw_rs (pure Rust — no C compilation required).
//
// Design:
//   • Index is built once at first KNN call from node_embeddings in graph.db.
//   • node_ids[i] maps HNSW label i → node_id stored in the graph DB.
//   • Staleness detection: on-disk count checked before each search; if it drifts
//     > REBUILD_THRESHOLD from the indexed count the index is rebuilt.
//
// BLOB format: 256 × f32 little-endian = 1024 bytes — matches model.rs output.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use hnsw_rs::prelude::*;
use rusqlite::Connection;

const REBUILD_THRESHOLD: f64 = 0.05; // rebuild when count drifts > 5 %

// HNSW construction parameters.
const MAX_NB_CONN: usize = 16; // M parameter
const NB_LAYER_MAX: usize = 16; // log₂(10M) ≈ 23; 16 is fine for < 1M nodes
const EF_CONSTRUCTION: usize = 400;

pub struct VecIndex {
    // 'static because insert() clones data into Vec<T> (PointData::V variant),
    // so no external slice references are held by the HNSW graph.
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// hnsw label i → on-disk node_id
    node_ids: Vec<i64>,
    on_disk_count: usize,
    db_path: PathBuf,
    model_id: String,
}

impl VecIndex {
    /// Build a fresh HNSW index from `node_embeddings` in `db_path`.
    pub fn build(db_path: &Path, model_id: &str) -> Result<Self> {
        let rows = load_embeddings(db_path, model_id)?;
        let n = rows.len();

        let hnsw = Hnsw::<f32, DistCosine>::new(
            MAX_NB_CONN,
            n.max(1),
            NB_LAYER_MAX,
            EF_CONSTRUCTION,
            DistCosine,
        );

        let mut node_ids = Vec::with_capacity(n);
        for (idx, (node_id, blob)) in rows.iter().enumerate() {
            let vec = crate::model::blob_to_f32(blob);
            hnsw.insert((&vec, idx));
            node_ids.push(*node_id);
        }

        tracing::info!(count = n, model_id, "HNSW index built");
        Ok(Self {
            hnsw,
            node_ids,
            on_disk_count: n,
            db_path: db_path.to_path_buf(),
            model_id: model_id.to_string(),
        })
    }

    pub fn count(&self) -> usize {
        self.node_ids.len()
    }

    /// K-nearest-neighbour search.
    /// `query_blob`: 1024-byte LE f32 blob matching dim=256.
    /// Returns up to `k` (node_id, cosine_distance) pairs.
    pub fn knn(&mut self, query_blob: &[u8], k: u32) -> Result<Vec<(i64, f32)>> {
        self.maybe_rebuild()?;

        if self.node_ids.is_empty() {
            return Ok(vec![]);
        }

        let query = crate::model::blob_to_f32(query_blob);
        let ef = (k as usize * 2).max(50);
        let neighbors = self.hnsw.search(query.as_slice(), k as usize, ef);

        let results = neighbors
            .iter()
            .filter_map(|n| {
                let node_id = *self.node_ids.get(n.d_id)?;
                Some((node_id, n.distance))
            })
            .collect();
        Ok(results)
    }

    fn maybe_rebuild(&mut self) -> Result<()> {
        let conn = Connection::open(&self.db_path)
            .context("open graph.db for staleness check")?;
        let current: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM node_embeddings WHERE model_id = ?1",
                [&self.model_id],
                |r| r.get(0),
            )
            .unwrap_or(self.on_disk_count as i64);
        drop(conn);

        let drift = (current as f64 - self.on_disk_count as f64).abs()
            / (self.on_disk_count.max(1) as f64);
        if drift > REBUILD_THRESHOLD {
            tracing::info!(
                old = self.on_disk_count,
                new = current,
                "embedding count drifted >{:.0}% — rebuilding HNSW index",
                REBUILD_THRESHOLD * 100.0,
            );
            let db_path = self.db_path.clone();
            let model_id = self.model_id.clone();
            *self = Self::build(&db_path, &model_id)?;
        }
        Ok(())
    }
}

fn load_embeddings(db_path: &Path, model_id: &str) -> Result<Vec<(i64, Vec<u8>)>> {
    let conn = Connection::open(db_path).context("open graph.db")?;
    let mut stmt = conn
        .prepare("SELECT node_id, embedding FROM node_embeddings WHERE model_id = ?1")
        .context("prepare node_embeddings query")?;
    let rows: Vec<(i64, Vec<u8>)> = stmt
        .query_map([model_id], |r| Ok((r.get(0)?, r.get(1)?)))
        .context("query node_embeddings")?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}
