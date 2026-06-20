// Disk-persisted HNSW index via usearch 2.x for O(log n) KNN over node embeddings.
//
// Design (replaces hnsw_rs in-memory index — RFC-018 Option B):
//   • Index lives on disk at ~/.travsr/models/<model-id>/hnsw.usearch.
//   • Daemon startup: load() reads the file once; no rebuild on process start.
//   • Reindex: new nodes are add()-ed incrementally; save() writes the updated file.
//   • Staleness detection: one stat() syscall per KNN call instead of
//     Connection::open() + SELECT COUNT(*) + drop(conn) (~5 ms → ~200 ns).
//
// BLOB format: 256 × f32 little-endian = 1024 bytes (matches model.rs output).

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context as _, Result};
use rusqlite::Connection;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub struct VecIndex {
    inner: Index,
    index_path: PathBuf,
    last_modified: SystemTime,
}

impl VecIndex {
    /// Load an existing index from disk. Returns None if the file does not exist yet.
    /// Call this from NomicPlugin::load() at daemon startup and in the incremental
    /// reindex path when hnsw.usearch already exists.
    pub fn try_load(index_path: &Path) -> Result<Option<Self>> {
        if !index_path.exists() {
            return Ok(None);
        }
        let inner = Index::new(&make_options()).context("create usearch Index")?;
        let path_str = index_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("index path is not valid UTF-8"))?;
        inner
            .load(path_str)
            .context("load usearch index from disk")?;
        let last_modified = std::fs::metadata(index_path)
            .context("stat index file")?
            .modified()
            .context("index file mtime")?;
        tracing::info!(
            count = inner.size(),
            path = %index_path.display(),
            "HNSW index loaded"
        );
        Ok(Some(Self {
            inner,
            index_path: index_path.to_path_buf(),
            last_modified,
        }))
    }

    /// Create an empty writable index. Used for the first reindex run when
    /// no hnsw.usearch file exists yet. Reserves `capacity` slots upfront.
    pub fn new_empty(index_path: &Path, capacity: usize) -> Result<Self> {
        let inner = Index::new(&make_options()).context("create usearch Index")?;
        inner
            .reserve(capacity)
            .context("reserve initial capacity")?;
        Ok(Self {
            inner,
            index_path: index_path.to_path_buf(),
            last_modified: SystemTime::UNIX_EPOCH,
        })
    }

    /// Full rebuild by streaming node_embeddings from graph.db.
    /// Peak RAM: one BLOB (1024 bytes) at a time — no full materialisation.
    /// Used as a recovery path when hnsw.usearch is missing but node_embeddings
    /// is already populated (e.g., after accidental index deletion).
    pub fn build_from_db(
        db_path: &Path,
        model_id: &str,
        index_path: &Path,
        expected_count: usize,
    ) -> Result<Self> {
        let conn = Connection::open(db_path).context("open graph.db")?;

        let n: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM node_embeddings WHERE model_id = ?1",
                [model_id],
                |r| r.get(0),
            )
            .unwrap_or(expected_count);

        let inner = Index::new(&make_options()).context("create usearch Index")?;
        inner.reserve(n).context("reserve capacity")?;

        let mut stmt =
            conn.prepare("SELECT node_id, embedding FROM node_embeddings WHERE model_id = ?1")?;
        let mut rows = stmt.query([model_id])?;
        let mut count = 0usize;
        while let Some(row) = rows.next()? {
            let node_id: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let vec = crate::model::blob_to_f32(&blob);
            inner.add(node_id as u64, &vec).context("usearch add")?;
            count += 1;
        }

        let path_str = index_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("index path not UTF-8"))?;
        inner.save(path_str).context("save usearch index")?;
        let last_modified = std::fs::metadata(index_path)
            .context("stat saved index")?
            .modified()
            .context("mtime")?;

        tracing::info!(count, path = %index_path.display(), "HNSW index built from DB");
        Ok(Self {
            inner,
            index_path: index_path.to_path_buf(),
            last_modified,
        })
    }

    /// Add one node's embedding to the index. Called per-node during reindex.
    /// usearch handles internal synchronisation; takes &self.
    pub fn add(&self, node_id: i64, vec: &[f32]) -> Result<()> {
        self.inner.add(node_id as u64, vec).context("usearch add")
    }

    /// Persist the current index to self.index_path.
    pub fn save(&self) -> Result<()> {
        let path_str = self
            .index_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("index path not UTF-8"))?;
        self.inner.save(path_str).context("save usearch index")
    }

    /// K-nearest-neighbour search.
    ///
    /// Checks mtime before searching; if the file changed on disk (background
    /// reindex completed), reloads the index. One stat() syscall per call (~200 ns)
    /// replaces the old Connection::open() + COUNT(*) + drop(conn) (~5 ms).
    ///
    /// `query_blob`: 1024-byte LE f32 blob (dim=256).
    /// Returns up to `k` (node_id, cosine_distance) pairs.
    pub fn knn(&mut self, query_blob: &[u8], k: u32) -> Result<Vec<(i64, f32)>> {
        let mtime = std::fs::metadata(&self.index_path)
            .context("stat index file")?
            .modified()
            .context("index mtime")?;
        if mtime > self.last_modified {
            tracing::info!("index file updated — reloading");
            *self = Self::try_load(&self.index_path)?
                .ok_or_else(|| anyhow::anyhow!("index file vanished after mtime change"))?;
        }

        if self.inner.size() == 0 {
            return Ok(vec![]);
        }

        let query = crate::model::blob_to_f32(query_blob);
        let results = self
            .inner
            .search(&query, k as usize)
            .context("usearch search")?;

        Ok(results
            .keys
            .iter()
            .zip(results.distances.iter())
            .map(|(&key, &dist)| (key as i64, dist))
            .collect())
    }

    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.inner.size()
    }
}

fn make_options() -> IndexOptions {
    IndexOptions {
        dimensions: 256,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: 16,
        expansion_add: 128,
        expansion_search: 64,
        multi: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(seed: u32) -> Vec<f32> {
        let mut v: Vec<f32> = (0u32..256)
            .map(|i| {
                let x = seed
                    .wrapping_mul(1664525)
                    .wrapping_add(1013904223)
                    .wrapping_add(i.wrapping_mul(22695477));
                (x as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    #[test]
    fn build_save_load_knn_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("travsr_embed_idx_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.usearch");

        let idx = VecIndex::new_empty(&path, 100).unwrap();
        for i in 0u32..100 {
            idx.add(i as i64, &unit_vec(i)).unwrap();
        }
        idx.save().unwrap();

        let mut loaded = VecIndex::try_load(&path).unwrap().unwrap();
        assert_eq!(loaded.count(), 100);

        let query = unit_vec(42);
        let query_blob: Vec<u8> = query.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let results = loaded.knn(&query_blob, 5).unwrap();

        assert!(!results.is_empty(), "KNN must return at least one result");
        assert_eq!(
            results[0].0, 42,
            "top-1 must be the query vector itself (node 42)"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
