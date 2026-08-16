// Disk-persisted HNSW index via usearch 2.x for O(log n) KNN over node embeddings.
//
// Design (replaces hnsw_rs in-memory index — RFC-018 Option B):
//   • Index lives on disk at ~/.travsr/models/<model-id>/hnsw.usearch.
//   • Daemon startup: load() reads the file once; no rebuild on process start.
//   • Reindex: new nodes are add()-ed incrementally; save() writes the updated file.
//   • Staleness detection: one stat() syscall per KNN call instead of
//     Connection::open() + SELECT COUNT(*) + drop(conn) (~5 ms → ~200 ns).
//
// BLOB format: 384 × f32 little-endian = 1536 bytes (BGE-small CLS-384 output).

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context as _, Result};
use rusqlite::Connection;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// #391: single source of truth for "should this node be embedded / indexed?".
///
/// The daemon's `embed_text IS NOT NULL` is the authoritative eligibility signal.
/// Structural/noise kinds are still excluded, but a node the daemon explicitly
/// opted in (embed_text populated) is admitted regardless of kind — this is how
/// data-format file nodes (yaml/toml/json/xml, `kind='file'`) reach the HNSW
/// without also pulling in source-file nodes (`kind='file'`, `embed_text` NULL)
/// via the `build_node_text` fallback.
///
/// Every node-*selection* query (index build here, pending-node selection and the
/// lazy-embed FTS path in `main.rs`) references this one predicate so the policy
/// can never drift between the write path and the index-build path. Assumes the
/// `nodes` table is aliased `n`. Caller/callee sub-queries keep the bare
/// exclusion list — a file must never be listed as a caller.
///
/// #376 W1: a `doc-chunk` with NULL `embed_text` is *ineligible*. Its entire
/// retrieval value is its prose, and the candidate query's COALESCE fallback
/// would embed the heading trail and path alone — a lexical-match-only vector
/// wearing a semantic vector's clothes, permanent because candidacy is
/// presence-only. Excluding it keeps the node a candidate until real text
/// exists (the daemon regenerates `embed_text` before spawning a pass). Code
/// kinds keep the fallback: signature + path + callers/callees is a weak but
/// real signal. `travsr-store::embed_progress`'s KIND_FILTER mirrors this — if
/// the two drift, `travsr embed status` reports coverage the sidecar disagrees
/// with and the daemon's auto-spawn either never fires or never stops.
pub(crate) const NODE_ELIGIBLE: &str =
    "((n.kind NOT IN ('file', 'file-module', 'import', 'module', 'field', 'variable') \
      OR n.embed_text IS NOT NULL) \
      AND NOT (n.kind = 'doc-chunk' AND n.embed_text IS NULL))";

/// #376 Phase 2: which *index* a node's embedding belongs in, not whether it
/// gets embedded at all. `NODE_ELIGIBLE` (embedding eligibility, used by the
/// reindex/lazy-embed candidate queries in `main.rs`) correctly admits
/// `doc-chunk` nodes — they must be embedded like any other node. But the two
/// spaces are separate HNSW files with separate recall semantics (plan §4.1:
/// no cross-modal ranking), so the *index build* queries need a stricter,
/// mutually exclusive split: a doc-chunk node must never enter the code index
/// (the plan's "mirror exclusion") and nothing else may enter the doc index.
pub(crate) const CODE_SPACE_ELIGIBLE: &str =
    "(n.kind != 'doc-chunk' AND (n.kind NOT IN ('file', 'file-module', 'import', 'module', 'field', 'variable') \
      OR n.embed_text IS NOT NULL))";

pub(crate) const DOC_SPACE_ELIGIBLE: &str = "n.kind = 'doc-chunk'";

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
        // dim is read from the file by usearch on load() — pass 1 as a placeholder.
        let inner = Index::new(&make_options(1)).context("create usearch Index")?;
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
    pub fn new_empty(index_path: &Path, capacity: usize, dim: usize) -> Result<Self> {
        let inner = Index::new(&make_options(dim)).context("create usearch Index")?;
        inner
            .reserve(capacity)
            .context("reserve initial capacity")?;
        Ok(Self {
            inner,
            index_path: index_path.to_path_buf(),
            last_modified: SystemTime::UNIX_EPOCH,
        })
    }

    /// Full rebuild by streaming node_embeddings from embed.db.
    /// Peak RAM: one BLOB (1536 bytes) at a time — no full materialisation.
    /// Used as a recovery path when hnsw.usearch is missing but node_embeddings
    /// is already populated (e.g., after accidental index deletion).
    ///
    /// RFC-019: embeddings live in embed.db; graph.db holds nodes for the JOIN.
    /// embed.db is ATTACHed to the graph.db connection as "edb" so the kind-filter
    /// JOIN works across both files without loading everything into memory.
    /// `node_filter` selects which space this build populates —
    /// [`CODE_SPACE_ELIGIBLE`] or [`DOC_SPACE_ELIGIBLE`] (#376 Phase 2). The two
    /// are mutually exclusive by construction (see their doc comments), so
    /// calling this once per space with a shared `embed_db_path` always
    /// partitions the corpus correctly.
    pub fn build_from_db(
        db_path: &Path,
        embed_db_path: &Path,
        model_id: &str,
        index_path: &Path,
        expected_count: usize,
        dim: usize,
        node_filter: &str,
    ) -> Result<Self> {
        let conn = Connection::open(db_path).context("open graph.db")?;
        let embed_db_str = embed_db_path
            .to_str()
            .context("embed.db path is not valid UTF-8")?;
        conn.execute_batch(&format!("ATTACH DATABASE '{embed_db_str}' AS edb"))
            .context("attach embed.db")?;

        // Index eligibility: exclude only structural/noise kinds. Every remaining
        // embedded node is indexed so KNN can surface it as a seed.
        //
        // NOTE (regression fix): an earlier version additionally required an
        // incoming ref/call edge ("zero blast radius → skip"). On a real repo that
        // silently dropped ~76% of embedded nodes from the HNSW — any function with
        // no *recorded* caller: entry points, trait/impl methods, dead code, and
        // crucially any newly-edited symbol whose Phase B call edges haven't been
        // re-resolved yet (e.g. right after an edit + reindex). Excluded nodes can
        // never be returned by KNN, so get_context seeded on unrelated nodes and
        // returned garbage even for exact-name queries.
        //
        // Recall must not be filtered at index-build time. The blast-radius /
        // PPR-expandability preference belongs in seed *ranking* downstream, not in
        // what the index contains. Dropping the clause also makes a full rebuild
        // consistent with the incremental add() path, which never filtered.
        // #391: shared eligibility predicate — see NODE_ELIGIBLE for rationale.
        // #376 Phase 2: `node_filter` is CODE_SPACE_ELIGIBLE or DOC_SPACE_ELIGIBLE,
        // never the raw NODE_ELIGIBLE (which admits doc-chunk nodes into a would-be
        // code index — the plan's "mirror exclusion" bug).

        let n: usize = conn
            .query_row(
                &format!(
                    "SELECT COUNT(*) FROM edb.node_embeddings e \
                     JOIN nodes n ON n.id = e.node_id \
                     WHERE e.model_id = ?1 AND {node_filter}"
                ),
                [model_id],
                |r| r.get(0),
            )
            .unwrap_or(expected_count);

        let inner = Index::new(&make_options(dim)).context("create usearch Index")?;
        inner.reserve(n).context("reserve capacity")?;

        let mut stmt = conn.prepare(&format!(
            "SELECT e.node_id, e.embedding \
             FROM edb.node_embeddings e \
             JOIN nodes n ON n.id = e.node_id \
             WHERE e.model_id = ?1 AND {node_filter}",
        ))?;
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
    ///
    /// Skip-if-present: handles crash-recovery where HNSW was updated but the
    /// matching SQLite COMMIT was rolled back, leaving the key in HNSW without
    /// a node_embeddings row.  The embedding is identical (same node, same model),
    /// so skipping the add is correct; the caller still writes the DB row.
    pub fn add(&self, node_id: i64, vec: &[f32]) -> Result<()> {
        if self.inner.contains(node_id as u64) {
            return Ok(());
        }
        self.inner.add(node_id as u64, vec).context("usearch add")
    }

    /// Persist the current index to self.index_path atomically.
    ///
    /// Writes to a `.tmp` sibling first, then renames, so the daemon's
    /// mtime-triggered reload in `knn()` never reads a partially-written file.
    #[allow(dead_code)]
    pub fn save(&self) -> Result<()> {
        let tmp = self.index_path.with_extension("usearch.tmp");
        let tmp_str = tmp
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("index tmp path not UTF-8"))?;
        self.inner
            .save(tmp_str)
            .context("save usearch index to tmp")?;
        std::fs::rename(&tmp, &self.index_path).context("atomic rename usearch index")?;
        Ok(())
    }

    /// K-nearest-neighbour search.
    ///
    /// Checks mtime before searching; if the file changed on disk (background
    /// reindex completed), reloads the index. One stat() syscall per call (~200 ns)
    /// replaces the old Connection::open() + COUNT(*) + drop(conn) (~5 ms).
    ///
    /// `query_blob`: 1536-byte LE f32 blob (dim=384).
    /// Returns up to `k` (node_id, cosine_distance) pairs.
    pub fn knn(&mut self, query_blob: &[u8], k: u32) -> Result<Vec<(i64, f32)>> {
        let mtime = std::fs::metadata(&self.index_path)
            .context("stat index file")?
            .modified()
            .context("index mtime")?;
        if mtime > self.last_modified {
            tracing::info!("index file updated — reloading");
            // #736 C2: free the old graph BEFORE loading the new file. The
            // previous `*self = Self::try_load(...)` built the replacement
            // index while the old one was still alive — a transient 2× of the
            // index's full size, at exactly the moment a just-finished reindex
            // has already elevated memory. A reload that then fails leaves an
            // empty index rather than the stale one; the next knn() sees the
            // unchanged mtime is still newer than last_modified and retries.
            self.inner = Index::new(&make_options(1)).context("create usearch Index")?;
            let path_str = self
                .index_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("index path is not valid UTF-8"))?;
            self.inner
                .load(path_str)
                .context("reload usearch index from disk")?;
            self.last_modified = mtime;
            tracing::info!(count = self.inner.size(), "HNSW index reloaded");
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

    /// Reserve capacity for `total` elements (absolute, not additional).
    /// Call this after loading an existing index before inserting new vectors.
    pub fn reserve(&self, total: usize) -> Result<()> {
        self.inner
            .reserve(total)
            .context("reserve usearch capacity")
    }

    /// Current number of indexed vectors.
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.inner.size()
    }
}

fn make_options(dim: usize) -> IndexOptions {
    IndexOptions {
        dimensions: dim,
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

    const TEST_DIM: usize = 384;

    fn unit_vec(seed: u32) -> Vec<f32> {
        let mut v: Vec<f32> = (0u32..TEST_DIM as u32)
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

        let idx = VecIndex::new_empty(&path, 100, TEST_DIM).unwrap();
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

    /// #391: the shared NODE_ELIGIBLE predicate must admit a data-format file node
    /// the daemon opted in (embed_text set), keep out a source-file node
    /// (kind='file', embed_text NULL), keep out structural noise (import), and
    /// still admit ordinary symbol kinds regardless of embed_text.
    #[test]
    fn node_eligible_admits_file_with_embed_text_only() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE nodes (id INTEGER PRIMARY KEY, kind TEXT, embed_text TEXT);
             INSERT INTO nodes (id, kind, embed_text) VALUES
                 (1, 'file',     'workflow: ci.yml'),  -- data-format file → IN
                 (2, 'file',     NULL),                -- source file      → OUT
                 (3, 'function', NULL),                -- symbol           → IN
                 (4, 'import',   NULL),                -- structural noise → OUT
                 (5, 'package',  'crate: travsr');     -- opted-in         → IN",
        )
        .unwrap();

        let sql = format!("SELECT id FROM nodes n WHERE {NODE_ELIGIBLE} ORDER BY id");
        let mut stmt = conn.prepare(&sql).unwrap();
        let ids: Vec<i64> = stmt
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(
            ids,
            vec![1, 3, 5],
            "eligible set must be {{file+embed_text, symbol, package+embed_text}}"
        );
    }

    /// #376 W1: a doc-chunk without prose must never be embedded. The candidate
    /// query's COALESCE fallback would build its text from the heading trail and
    /// path alone, and presence-only candidacy would then make that degraded
    /// vector permanent — invisible in the output, since the docs section prints
    /// no score. It stays a candidate until the daemon regenerates `embed_text`.
    #[test]
    fn node_eligible_rejects_doc_chunk_without_prose() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE nodes (id INTEGER PRIMARY KEY, kind TEXT, embed_text TEXT);
             INSERT INTO nodes (id, kind, embed_text) VALUES
                 (1, 'doc-chunk', 'doc: readme > setup | install it'), -- IN
                 (2, 'doc-chunk', NULL),                               -- OUT
                 (3, 'function',  NULL);                               -- IN (fallback ok)",
        )
        .unwrap();

        let sql = format!("SELECT id FROM nodes n WHERE {NODE_ELIGIBLE} ORDER BY id");
        let mut stmt = conn.prepare(&sql).unwrap();
        let ids: Vec<i64> = stmt
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(
            ids,
            vec![1, 3],
            "a doc-chunk with NULL embed_text must be ineligible"
        );
    }

    /// #376 Phase 2: CODE_SPACE_ELIGIBLE and DOC_SPACE_ELIGIBLE must partition
    /// the corpus with no overlap. In particular a doc-chunk node — which
    /// always has `embed_text` set (plan §3.2) and would therefore pass the
    /// plain NODE_ELIGIBLE check — must be excluded from the code space (the
    /// plan's "mirror exclusion") while still being admitted to the doc space.
    #[test]
    fn code_and_doc_space_eligible_partition_with_no_overlap() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE nodes (id INTEGER PRIMARY KEY, kind TEXT, embed_text TEXT);
             INSERT INTO nodes (id, kind, embed_text) VALUES
                 (1, 'file',      'workflow: ci.yml'), -- data-format file → code
                 (2, 'file',      NULL),               -- source file      → neither
                 (3, 'function',  NULL),                -- symbol           → code
                 (4, 'import',    NULL),                -- structural noise → neither
                 (5, 'doc-chunk', 'doc: intro'),         -- doc chunk        → docs
                 (6, 'doc-chunk', 'doc: another section'); -- doc chunk      → docs",
        )
        .unwrap();

        let code_sql = format!("SELECT id FROM nodes n WHERE {CODE_SPACE_ELIGIBLE} ORDER BY id");
        let mut stmt = conn.prepare(&code_sql).unwrap();
        let code_ids: Vec<i64> = stmt
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(
            code_ids,
            vec![1, 3],
            "code space must exclude doc-chunk nodes even though they carry embed_text"
        );

        let doc_sql = format!("SELECT id FROM nodes n WHERE {DOC_SPACE_ELIGIBLE} ORDER BY id");
        let mut stmt = conn.prepare(&doc_sql).unwrap();
        let doc_ids: Vec<i64> = stmt
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(doc_ids, vec![5, 6]);

        assert!(
            code_ids.iter().all(|id| !doc_ids.contains(id)),
            "code and doc spaces must never share a node id"
        );
    }

    /// #376 Phase 2: `build_from_db` called once per space against a shared
    /// embed.db must produce two separate on-disk indices, each containing
    /// only its own space's vectors — the end-to-end version of the predicate
    /// test above, through the real ATTACH + stream + save path.
    #[test]
    fn build_from_db_partitions_code_and_doc_spaces() {
        let dir = std::env::temp_dir().join(format!(
            "travsr_embed_space_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("graph.db");
        let embed_db_path = dir.join("embed.db");
        let code_index_path = dir.join("code.usearch");
        let doc_index_path = dir.join("docs.usearch");

        // graph.db: 2 code-eligible nodes, 2 doc-chunk nodes.
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE nodes (id INTEGER PRIMARY KEY, kind TEXT, embed_text TEXT);
             INSERT INTO nodes (id, kind, embed_text) VALUES
                 (1, 'function',  NULL),
                 (2, 'function',  NULL),
                 (3, 'doc-chunk', 'doc: intro'),
                 (4, 'doc-chunk', 'doc: another section');",
        )
        .unwrap();
        drop(conn);

        // embed.db: one embedding row per node, same model_id.
        let embed_conn = Connection::open(&embed_db_path).unwrap();
        embed_conn
            .execute_batch(
                "CREATE TABLE node_embeddings (
                     node_id INTEGER NOT NULL, model_id TEXT NOT NULL, embedding BLOB NOT NULL,
                     PRIMARY KEY (node_id, model_id)
                 ) WITHOUT ROWID;",
            )
            .unwrap();
        for id in 1i64..=4 {
            let blob: Vec<u8> = unit_vec(id as u32)
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            embed_conn
                .execute(
                    "INSERT INTO node_embeddings (node_id, model_id, embedding) VALUES (?1, 'm', ?2)",
                    rusqlite::params![id, blob],
                )
                .unwrap();
        }
        drop(embed_conn);

        VecIndex::build_from_db(
            &db_path,
            &embed_db_path,
            "m",
            &code_index_path,
            2,
            TEST_DIM,
            CODE_SPACE_ELIGIBLE,
        )
        .unwrap();
        VecIndex::build_from_db(
            &db_path,
            &embed_db_path,
            "m",
            &doc_index_path,
            2,
            TEST_DIM,
            DOC_SPACE_ELIGIBLE,
        )
        .unwrap();

        let code_idx = VecIndex::try_load(&code_index_path).unwrap().unwrap();
        assert_eq!(
            code_idx.count(),
            2,
            "code index must contain only nodes 1,2"
        );
        let doc_idx = VecIndex::try_load(&doc_index_path).unwrap().unwrap();
        assert_eq!(doc_idx.count(), 2, "doc index must contain only nodes 3,4");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
