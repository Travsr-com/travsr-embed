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
use std::time::{Instant, SystemTime};

use anyhow::{Context as _, Result};
use rusqlite::Connection;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// Minimum spacing between reload attempts in [`VecIndex::knn`] (travsr#735
/// follow-up). Two distinct pathologies collapse into one gate:
///
/// 1. A reload that keeps FAILING (partial/corrupt index file left by a killed
///    reindex) used to be retried on every single KNN call, forever, each
///    attempt constructing a fresh native usearch index. On a busy daemon that
///    is an allocation loop with no exit.
/// 2. A reload that keeps SUCCEEDING (an active reindex repeatedly publishing
///    the file) used to re-map the index per query; serving a few seconds
///    stale is harmless, re-mapping per query is not.
///
/// Five seconds keeps the serving index near-fresh (the file only changes when
/// a whole rebuild completes) while bounding both loops to at most one native
/// index construction per interval.
const RELOAD_MIN_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// Pure decision for "may knn() attempt a reload now?" — split out of
/// [`VecIndex::knn`] so the throttle is testable without a usearch file.
fn reload_due(mtime_changed: bool, last_attempt: Option<Instant>, now: Instant) -> bool {
    mtime_changed && last_attempt.is_none_or(|t| now.duration_since(t) >= RELOAD_MIN_INTERVAL)
}

/// Smallest on-disk size a legitimately published index can have.
///
/// usearch's native `view()`/`load()` parse the file header without
/// validation and can SIGSEGV on truncated bytes (observed under test on
/// Linux), so obviously-degenerate files must be rejected in Rust before any
/// native parsing. Every published code/doc index holds at least one
/// f32-384 vector (1536 bytes of data alone) and is written via
/// save-to-tmp + rename, so a smaller file is always truncation residue from
/// a killed process, never a valid index. This floor cannot catch LARGE
/// garbage; that residual risk is bounded by the host's respawn cap.
const MIN_PLAUSIBLE_INDEX_BYTES: u64 = 512;

/// Cheap Rust-side sanity gate run before handing a file to native usearch
/// parsing (travsr#735 follow-up). Errors on files too small to be a real
/// index; the caller decides whether to quarantine (write path) or keep
/// serving the previous index (serve path).
fn plausible_index_file(index_path: &Path) -> Result<()> {
    let len = std::fs::metadata(index_path)
        .with_context(|| format!("stat {}", index_path.display()))?
        .len();
    anyhow::ensure!(
        len >= MIN_PLAUSIBLE_INDEX_BYTES,
        "index file {} is {} bytes, below the {} byte minimum any published \
         index can have; treating as truncated residue from an interrupted run",
        index_path.display(),
        len,
        MIN_PLAUSIBLE_INDEX_BYTES
    );
    Ok(())
}

/// Move a corrupt index file aside as `<name>.corrupt` so the next open starts
/// from a clean slate instead of failing on the same bytes forever (travsr#735
/// follow-up: nothing ever cleaned a corrupt index, so a reindex that died
/// mid-save poisoned every subsequent run). Keeps only the most recent
/// quarantined copy. Falls back to deleting the file when the rename itself
/// fails (e.g. a stale `.corrupt` on a read-only sibling). Returns the
/// quarantine path when the file was moved.
///
/// Write-path only: the serving (read) path must never delete a file a
/// concurrent reindex may be about to replace.
pub(crate) fn quarantine_corrupt_index(index_path: &Path) -> Option<PathBuf> {
    let quarantined = index_path.with_extension("usearch.corrupt");
    let _ = std::fs::remove_file(&quarantined);
    match std::fs::rename(index_path, &quarantined) {
        Ok(()) => {
            tracing::warn!(
                from = %index_path.display(),
                to = %quarantined.display(),
                "quarantined corrupt HNSW index; a fresh index will be rebuilt"
            );
            Some(quarantined)
        }
        Err(rename_err) => {
            match std::fs::remove_file(index_path) {
                Ok(()) => tracing::warn!(
                    path = %index_path.display(),
                    "deleted corrupt HNSW index (quarantine rename failed: {rename_err})"
                ),
                Err(rm_err) => tracing::warn!(
                    path = %index_path.display(),
                    "could not quarantine ({rename_err}) or delete ({rm_err}) corrupt HNSW index"
                ),
            }
            None
        }
    }
}

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
    /// True when `inner` is a memory-mapped `view()` rather than a `load()`
    /// copy (#736 item 4). A viewed index is read-only: `add` becomes a
    /// no-op and reloads re-view. Serving uses views so the OS can evict
    /// index pages under memory pressure instead of the daemon holding the
    /// whole graph in anonymous memory.
    viewed: bool,
    /// When [`Self::knn`] last attempted an mtime-triggered reload, successful
    /// or not. Gates reload attempts to [`RELOAD_MIN_INTERVAL`] so neither a
    /// persistently corrupt file nor a rapidly republished one turns the KNN
    /// path into a per-query native-index construction loop (travsr#735
    /// follow-up).
    last_reload_attempt: Option<Instant>,
}

impl VecIndex {
    /// Load an existing index from disk. Returns None if the file does not exist yet.
    /// Call this in the incremental reindex path when hnsw.usearch already
    /// exists — reindex must `add()`, which a viewed index cannot.
    /// The daemon serving path uses [`Self::try_serve`] instead.
    pub fn try_load(index_path: &Path) -> Result<Option<Self>> {
        if !index_path.exists() {
            return Ok(None);
        }
        // Reject truncation residue before native parsing (see
        // MIN_PLAUSIBLE_INDEX_BYTES); the reindex caller quarantines on Err.
        plausible_index_file(index_path)?;
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
            viewed: false,
            last_reload_attempt: None,
        }))
    }

    /// Open an existing index for SERVING (#736 item 4). Returns None if the
    /// file does not exist yet.
    ///
    /// On Linux/macOS this memory-maps the file (`usearch::view`) instead of
    /// copying it into RAM: a 264k-node repo's index is hundreds of MB, and a
    /// view lets the OS page it in on demand and evict it under memory
    /// pressure — the daemon's resident footprint no longer includes the whole
    /// graph. The trade: a viewed index is immutable, so the lazy-embed path's
    /// `add()` becomes a no-op (see [`Self::add`] — the vector is still
    /// persisted to embed.db and enters the index on the next reindex).
    ///
    /// On Windows this falls back to `try_load` (a RAM copy): holding a file
    /// mapping there takes a sharing lock, and the reindex sidecar's
    /// `build_from_db` save would fail with a sharing violation while the
    /// daemon serves — a broken reindex is worse than the memory win.
    pub fn try_serve(index_path: &Path) -> Result<Option<Self>> {
        #[cfg(windows)]
        {
            Self::try_load(index_path)
        }
        #[cfg(not(windows))]
        {
            if !index_path.exists() {
                return Ok(None);
            }
            // Reject truncation residue before native parsing (see
            // MIN_PLAUSIBLE_INDEX_BYTES).
            plausible_index_file(index_path)?;
            let inner = Index::new(&make_options(1)).context("create usearch Index")?;
            let path_str = index_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("index path is not valid UTF-8"))?;
            inner.view(path_str).context("view (mmap) usearch index")?;
            let last_modified = std::fs::metadata(index_path)
                .context("stat index file")?
                .modified()
                .context("index file mtime")?;
            tracing::info!(
                count = inner.size(),
                path = %index_path.display(),
                "HNSW index viewed (mmap)"
            );
            Ok(Some(Self {
                inner,
                index_path: index_path.to_path_buf(),
                last_modified,
                viewed: true,
                last_reload_attempt: None,
            }))
        }
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
            viewed: false,
            last_reload_attempt: None,
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

        // #18 review (blocking): never save onto the live index path in place.
        // The serving daemon may hold an mmap view() of that inode (try_serve),
        // and truncating + rewriting it underneath the mapping is a SIGBUS
        // (touching a page past the momentarily-truncated length) or torn
        // reads (usearch walking a half-written graph) — the danger window is
        // the whole write, not just a moment. Write a sibling tmp and rename:
        // the daemon's existing mapping keeps the OLD inode alive and intact
        // until its next mtime-triggered re-view, the same protocol as
        // [`VecIndex::save`]. (On Windows the daemon serves from a load() copy,
        // so rename-over-open-file semantics are never exercised there.)
        let tmp = index_path.with_extension("usearch.tmp");
        let tmp_str = tmp
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("index tmp path not UTF-8"))?;
        inner.save(tmp_str).context("save usearch index to tmp")?;
        std::fs::rename(&tmp, index_path).context("atomic rename usearch index")?;
        let last_modified = std::fs::metadata(index_path)
            .context("stat saved index")?
            .modified()
            .context("mtime")?;

        tracing::info!(count, path = %index_path.display(), "HNSW index built from DB");
        Ok(Self {
            inner,
            index_path: index_path.to_path_buf(),
            last_modified,
            viewed: false,
            last_reload_attempt: None,
        })
    }

    /// Add one node's embedding to the index. Called per-node during reindex.
    /// usearch handles internal synchronisation; takes &self.
    ///
    /// Skip-if-present: handles crash-recovery where HNSW was updated but the
    /// matching SQLite COMMIT was rolled back, leaving the key in HNSW without
    /// a node_embeddings row.  The embedding is identical (same node, same model),
    /// so skipping the add is correct; the caller still writes the DB row.
    ///
    /// On a viewed (mmap) serving index the add is a no-op (#736 item 4): the
    /// mapping is read-only. The only caller that adds to a serving index is
    /// the lazy-embed path, which already treats the add as best-effort — the
    /// vector is persisted to embed.db and enters the index on the next
    /// reindex/reload, and the current query scores it directly by cosine.
    pub fn add(&self, node_id: i64, vec: &[f32]) -> Result<()> {
        if self.viewed {
            tracing::debug!(
                node_id,
                "viewed index is immutable — lazy add deferred to next reindex"
            );
            return Ok(());
        }
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
        // travsr#735 follow-up: reload attempts are throttled (see
        // RELOAD_MIN_INTERVAL). Unthrottled, a persistently failing reload was
        // retried on every KNN call forever, and a rapidly republished file
        // was re-mapped per query — both are native-index construction loops.
        if reload_due(
            mtime > self.last_modified,
            self.last_reload_attempt,
            Instant::now(),
        ) {
            self.last_reload_attempt = Some(Instant::now());
            // Reject truncation residue BEFORE freeing or replacing anything:
            // native usearch parsing can crash on such bytes, and even a clean
            // failure would cost the current index on the load() branch. Keep
            // whatever is being served (search proceeds below on the current
            // index) and retry after the backoff.
            if let Err(e) = plausible_index_file(&self.index_path) {
                tracing::warn!(
                    "updated index file looks like truncation residue; keeping \
                     the current index and retrying after backoff: {e}"
                );
            } else {
                self.reload_from_disk(mtime)?;
            }
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

    /// Replace `inner` from the on-disk file after an mtime change; called
    /// only through the throttled, plausibility-checked gate in [`Self::knn`].
    fn reload_from_disk(&mut self, mtime: SystemTime) -> Result<()> {
        tracing::info!("index file updated — reloading");
        let path_str = self
            .index_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("index path is not valid UTF-8"))?;
        // A serving handle re-views (mmap), a reindex handle re-loads —
        // reload must never silently change the handle's mutability class.
        if self.viewed {
            // A view is a file mapping, not a RAM copy, so building the
            // replacement before dropping the old one costs address space
            // only — and on failure the daemon KEEPS SERVING the previous
            // index instead of degrading to an empty one (travsr#735
            // follow-up; the file is published by atomic rename, so a
            // failed view means a genuinely bad file, not a torn write).
            let fresh = Index::new(&make_options(1)).context("create usearch Index")?;
            match fresh.view(path_str) {
                Ok(()) => {
                    self.inner = fresh;
                    self.last_modified = mtime;
                    tracing::info!(count = self.inner.size(), "HNSW index reloaded");
                }
                Err(e) => {
                    tracing::warn!(
                        path = %self.index_path.display(),
                        "re-view of updated HNSW index failed; keeping the \
                         previous index and retrying after backoff: {e}"
                    );
                }
            }
        } else {
            // #736 C2: a load() is a RAM copy, so free the old graph BEFORE
            // loading the new file — building the replacement while the old
            // one is alive is a transient 2x of the index's full size, at
            // exactly the moment a just-finished reindex has already
            // elevated memory. A reload that then fails leaves an empty
            // index; the throttle in knn() bounds how often it is retried.
            self.inner = Index::new(&make_options(1)).context("create usearch Index")?;
            self.inner
                .load(path_str)
                .context("reload usearch index from disk")?;
            self.last_modified = mtime;
            tracing::info!(count = self.inner.size(), "HNSW index reloaded");
        }
        Ok(())
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

    /// Whether this handle is a read-only mmap view (see [`Self::try_serve`]).
    ///
    /// #18 review: `add` on a viewed handle is a silent `Ok(())` no-op, which
    /// is correct for the lazy-embed serving path but would be a disaster on
    /// the reindex path — the embed.db row would be written, `NOT EXISTS`
    /// would consider the node done, and the vector would never enter any
    /// index. The reindex paths assert against this at handle construction so
    /// the invariant is checked structurally, not by convention.
    pub fn is_viewed(&self) -> bool {
        self.viewed
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

    /// travsr#735 follow-up: the reload throttle must fire on the first
    /// changed-mtime observation, then hold further attempts inside the
    /// interval, then release, and must never fire when the mtime is unchanged.
    #[test]
    fn reload_due_throttles_attempts() {
        let t0 = Instant::now();
        assert!(reload_due(true, None, t0), "first attempt is always due");
        assert!(
            !reload_due(true, Some(t0), t0 + std::time::Duration::from_millis(100)),
            "an attempt inside the interval must be held"
        );
        assert!(
            reload_due(true, Some(t0), t0 + RELOAD_MIN_INTERVAL),
            "an attempt at the interval boundary must be released"
        );
        assert!(
            !reload_due(false, None, t0),
            "an unchanged mtime never triggers a reload"
        );
    }

    /// travsr#735 follow-up: a corrupt index file must be moved aside so the
    /// next open starts fresh, and re-quarantining must replace the previous
    /// quarantined copy rather than failing.
    #[test]
    fn quarantine_moves_corrupt_file_aside() {
        let dir = std::env::temp_dir().join(format!(
            "travsr_embed_quarantine_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.hnsw.usearch");
        std::fs::write(&path, b"definitely not a usearch file").unwrap();

        let moved = quarantine_corrupt_index(&path).expect("rename must succeed");
        assert!(
            !path.exists(),
            "corrupt file must be gone from the live path"
        );
        assert!(moved.exists(), "quarantined copy must exist");
        assert_eq!(moved.extension().unwrap(), "corrupt");

        // A second corrupt file replaces the previous quarantined copy.
        std::fs::write(&path, b"newer garbage").unwrap();
        let moved2 = quarantine_corrupt_index(&path).expect("second rename must succeed");
        assert_eq!(moved, moved2);
        assert_eq!(std::fs::read(&moved2).unwrap(), b"newer garbage");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// travsr#735 follow-up: when a served index is replaced on disk by
    /// truncation residue, the daemon must keep answering from the previously
    /// held index instead of degrading to an empty one (or handing the bytes
    /// to native usearch parsing, which was observed to SIGSEGV on Linux),
    /// and must not retry on every call inside the backoff window. Runs on
    /// every platform: the plausibility gate fires before the view/load split.
    #[test]
    fn failed_review_keeps_serving_the_old_index() {
        let dir =
            std::env::temp_dir().join(format!("travsr_embed_keepold_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("serve.usearch");

        let idx = VecIndex::new_empty(&path, 10, TEST_DIM).unwrap();
        for i in 0u32..10 {
            idx.add(i as i64, &unit_vec(i)).unwrap();
        }
        idx.save().unwrap();

        let mut served = VecIndex::try_serve(&path).unwrap().unwrap();
        let query: Vec<u8> = unit_vec(3).iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(served.knn(&query, 3).unwrap()[0].0, 3);

        // Publish garbage over the index the way production does: an atomic
        // rename of a NEW inode. Overwriting in place would corrupt the old
        // mapping itself (same inode) — exactly what the rename protocol in
        // build_from_db/save exists to prevent. The sleep guarantees the new
        // file's mtime is strictly newer on coarse-granularity filesystems.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let garbage = dir.join("garbage.tmp");
        std::fs::write(&garbage, b"garbage that is not an index").unwrap();
        std::fs::rename(&garbage, &path).unwrap();

        // The reload is refused by the plausibility gate, and KNN must still
        // answer from the previously held index.
        let results = served
            .knn(&query, 3)
            .expect("knn must not error when the replacement file is rejected");
        assert_eq!(
            results[0].0, 3,
            "old index must keep serving after a rejected reload"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// travsr#735 follow-up: the write path's try_load must reject truncation
    /// residue with a clean Err (which the reindex caller quarantines), never
    /// hand it to native parsing.
    #[test]
    fn try_load_rejects_truncation_residue_cleanly() {
        let dir = std::env::temp_dir().join(format!(
            "travsr_embed_tiny_index_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.hnsw.usearch");
        std::fs::write(&path, b"way too small to be an index").unwrap();

        let err = match VecIndex::try_load(&path) {
            Ok(_) => panic!("tiny file must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("below the"),
            "error must explain the size floor, got: {err:#}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

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

    /// #736 item 4: the serving path must answer KNN through a view (mmap on
    /// unix; load-fallback on Windows) with the same results as a load, and a
    /// viewed index must silently no-op lazy adds instead of erroring.
    #[test]
    fn serve_knn_roundtrip_and_add_is_noop() {
        let dir = std::env::temp_dir().join(format!(
            "travsr_embed_serve_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("serve.usearch");

        let idx = VecIndex::new_empty(&path, 100, TEST_DIM).unwrap();
        for i in 0u32..100 {
            idx.add(i as i64, &unit_vec(i)).unwrap();
        }
        idx.save().unwrap();

        let mut served = VecIndex::try_serve(&path).unwrap().unwrap();
        assert_eq!(served.count(), 100);
        let query = unit_vec(7);
        let query_blob: Vec<u8> = query.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let results = served.knn(&query_blob, 5).unwrap();
        assert_eq!(results[0].0, 7, "top-1 must be the query vector itself");

        // Lazy add on a serving index must never PANIC. On a viewed (unix)
        // index it is an Ok no-op; on the Windows load-fallback usearch may
        // refuse the insert (capacity not reserved) — the production
        // lazy-embed path ignores exactly that error, so the contract here is
        // "non-fatal", not "succeeds".
        let _ = served.add(1_000, &unit_vec(1_000));
        #[cfg(not(windows))]
        assert_eq!(served.count(), 100, "viewed index must not grow");

        let _ = std::fs::remove_dir_all(&dir);
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
