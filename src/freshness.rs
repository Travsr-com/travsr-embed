//! Content-addressed embedding freshness (#376 W1/W3).
//!
//! Why this module exists: before it, an embedding was invalidated *only* by the
//! `node_tombstones` CDC log, and candidacy for (re-)embedding was presence-only
//! (`NOT EXISTS (SELECT 1 FROM node_embeddings …)`). Three consequences, all
//! measured on a live repo:
//!
//! * `reindex_replace` deletes every node under a changed path, so a one-word
//!   edit tombstoned — and therefore re-embedded — every chunk in the file.
//! * A tombstone that was never consumed (log pruned, or no pass ever ran)
//!   left a vector in `embed.db` and in the live HNSW that no longer matched
//!   its node's content, with nothing to detect it.
//! * A node embedded before its `embed_text` existed was embedded from a
//!   synthesized heading-only fallback and, being present, was never revisited.
//!
//! The fix is to make the *content* the invalidation key: every embedding row
//! carries `text_hash`, the hash of the exact text that produced the vector. A
//! vector is stale iff its stored hash differs from the hash of the node's
//! current text. Tombstones stop being the source of truth and become what they
//! always should have been — a cheap hint about which nodes are worth checking.
//!
//! Deliberate asymmetry between the two spaces:
//!
//! * **code**: verified only for tombstoned nodes. A full scan would hash every
//!   `embed_text` in the repo on every pass (~150 MB on kubernetes) to catch a
//!   case the CDC log already covers.
//! * **docs**: fully verified every pass. The doc corpus is 2-4 orders of
//!   magnitude smaller (~10³ chunks), and a degraded doc vector is invisible in
//!   the output (§4.1 prints no score), so cheap total verification is the only
//!   honest guarantee.
//!
//! A node whose `embed_text` is currently NULL is never invalidated by hash: its
//! true text is not knowable yet (a fresh `travsr init` NULLs the column for the
//! whole repo until the daemon or CLI regenerates it), and deleting the vector
//! there would re-embed from the degraded fallback — the very bug this module
//! closes. Those tombstones are *deferred*, not acked, so a later pass re-checks
//! them once real text exists.

use anyhow::{Context as _, Result};
use rusqlite::Connection;

/// Doc-space rows are fully verified per pass only up to this many rows. Above
/// it the doc space is treated like the code space (tombstone-driven only), so
/// a pathological doc corpus can never make a pass unboundedly expensive.
const DOC_VERIFY_MAX_ROWS: i64 = 200_000;

/// 128-bit FNV-1a of the exact text handed to the encoder.
///
/// Not a security primitive — the only requirement is that a content change
/// changes the digest. FNV-1a is chosen over sha2/blake3 because it needs no
/// dependency (this sidecar's dependency set is audited per release) and costs
/// ~1 ns/byte, i.e. microseconds for a full doc corpus.
pub(crate) fn text_hash(text: &str) -> String {
    const OFFSET_BASIS: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    let mut h = OFFSET_BASIS;
    for b in text.as_bytes() {
        h ^= *b as u128;
        h = h.wrapping_mul(PRIME);
    }
    format!("{h:032x}")
}

/// What a single invalidation pass did. Reported by the caller, and asserted on
/// in tests — the counters are the only externally visible proof that a pass
/// distinguished "changed" from "merely re-indexed".
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InvalidationReport {
    /// Tombstoned nodes whose content hash still matched: vector kept.
    pub(crate) unchanged: usize,
    /// Embedding rows deleted because their content hash no longer matches.
    pub(crate) stale: usize,
    /// Embedding rows deleted because the node itself is gone.
    pub(crate) orphaned: usize,
    /// Tombstones left unacked because the node's `embed_text` is NULL — its
    /// current text is not knowable yet.
    pub(crate) deferred: usize,
    /// Tombstones acked without work because the node has no vector at all
    /// (`import`, `file-module` and friends are never embedded). Measured at
    /// 374 of 374 pending tombstones on this repo, so without this case the
    /// backlog would never reach zero on any repo.
    pub(crate) not_embedded: usize,
    /// Doc-space rows deleted by the full verification sweep.
    pub(crate) doc_stale: usize,
}

impl InvalidationReport {
    /// True when this pass removed at least one vector, i.e. the on-disk HNSW
    /// index no longer matches `embed.db` and must be rebuilt even if there is
    /// nothing new to embed.
    pub(crate) fn removed_any(&self) -> bool {
        self.stale > 0 || self.orphaned > 0 || self.doc_stale > 0
    }
}

/// Ensure `edb.node_embeddings` exists and carries the `text_hash` column.
///
/// `ALTER TABLE … ADD COLUMN` on an existing index is the migration: legacy rows
/// get NULL, which [`apply_invalidation`] treats as "unverifiable, therefore
/// stale" the first time it looks at that row.
pub(crate) fn ensure_schema(conn: &Connection, qualifier: &str) -> Result<()> {
    conn.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS {qualifier}node_embeddings (
             node_id   INTEGER NOT NULL,
             model_id  TEXT    NOT NULL,
             embedding BLOB    NOT NULL,
             text_hash TEXT,
             PRIMARY KEY (node_id, model_id)
         ) WITHOUT ROWID;
         CREATE INDEX IF NOT EXISTS {qualifier}idx_node_embeddings_model
             ON node_embeddings(model_id);"
    ))
    .context("create embed.db schema")?;

    // pragma_table_info's second argument is the schema name — without it the
    // probe would inspect `main` while the table lives in the attached `edb`.
    let schema = qualifier.trim_end_matches('.');
    let probe = if schema.is_empty() {
        "SELECT COUNT(*) FROM pragma_table_info('node_embeddings') WHERE name = 'text_hash'"
            .to_string()
    } else {
        format!(
            "SELECT COUNT(*) FROM pragma_table_info('node_embeddings', '{schema}') \
             WHERE name = 'text_hash'"
        )
    };
    let has_hash: bool = conn
        .prepare(&probe)
        .and_then(|mut s| s.query_row([], |r| r.get::<_, i64>(0)))
        .map(|n| n > 0)
        .unwrap_or(false);
    if !has_hash {
        conn.execute_batch(&format!(
            "ALTER TABLE {qualifier}node_embeddings ADD COLUMN text_hash TEXT"
        ))
        .context("add text_hash column to node_embeddings")?;
        tracing::info!("migrated embed.db: node_embeddings.text_hash added");
    }
    Ok(())
}

/// Verify and invalidate embeddings, then (optionally) ack the CDC log.
///
/// Runs in one transaction over `graph.db` + attached `edb`:
///
/// 1. **orphan sweep** — any embedding whose node no longer exists is dropped.
///    Catches deletions whose tombstone was lost (log pruned by the 7-day /
///    50 k-row GC, or acked by a pass that crashed before the HNSW rebuild).
/// 2. **CDC verification** — for each tombstoned node that still exists and has
///    `embed_text`, delete only the embedding rows whose `text_hash` differs.
///    Unchanged chunks of an edited file keep their vectors: this is what turns
///    "edit one section, re-embed the whole file" into "re-embed one section".
/// 3. **doc-space sweep** — every doc-chunk row is hash-checked regardless of
///    tombstones (see module docs).
/// 4. **ack** — resolved tombstones are deleted; deferred ones are kept.
///
/// `ack = false` is used by the pure `--rebuild-index` path, which can delete a
/// stale vector but cannot re-embed it: keeping the tombstone lets the next real
/// pass do that work.
pub(crate) fn apply_invalidation(
    conn: &Connection,
    ack: bool,
    verify_docs: bool,
) -> Result<InvalidationReport> {
    let mut report = InvalidationReport::default();

    conn.execute_batch("BEGIN IMMEDIATE")
        .context("begin invalidation transaction")?;
    let result = (|| -> Result<()> {
        // ── 1. orphan sweep ──────────────────────────────────────────────────
        report.orphaned = conn
            .execute(
                "DELETE FROM edb.node_embeddings \
                 WHERE node_id NOT IN (SELECT id FROM nodes)",
                [],
            )
            .context("orphan embedding sweep")?;

        // ── 2. CDC verification ──────────────────────────────────────────────
        // The digest is computed inside the row mapper rather than collected as
        // text: a repo-wide re-index tombstones every node, so materialising
        // `embed_text` for the whole backlog would hold the repo's entire corpus
        // (~150 MB on kubernetes) in memory inside a write transaction. A
        // 32-char digest per row bounds this at ~kilobytes regardless of repo
        // size. `None` still means "text not knowable yet" — see step 2's match.
        let pending: Vec<(i64, Option<String>, bool)> = {
            let mut stmt = conn
                .prepare(
                    "SELECT t.node_id, n.embed_text, \
                            EXISTS (SELECT 1 FROM edb.node_embeddings e \
                                    WHERE e.node_id = t.node_id) \
                     FROM (SELECT DISTINCT node_id FROM node_tombstones) t \
                     JOIN nodes n ON n.id = t.node_id",
                )
                .context("prepare tombstone verification query")?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, Option<String>>(1)?.as_deref().map(text_hash),
                        r.get::<_, bool>(2)?,
                    ))
                })
                .context("run tombstone verification query")?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .context("decode tombstone verification rows")?
        };

        let mut resolved: Vec<i64> = Vec::with_capacity(pending.len());
        {
            let mut del = conn
                .prepare(
                    "DELETE FROM edb.node_embeddings \
                     WHERE node_id = ?1 AND (text_hash IS NULL OR text_hash <> ?2)",
                )
                .context("prepare stale-vector delete")?;
            for (node_id, text_digest, has_vector) in &pending {
                // Nothing to invalidate: the node was never embedded (an
                // `import` or `file-module` node, or one still queued). Acked
                // immediately — deferring these would pin the backlog above
                // zero forever, since no future pass will ever give them one.
                if !has_vector {
                    report.not_embedded += 1;
                    resolved.push(*node_id);
                    continue;
                }
                match text_digest {
                    // Text not knowable yet — keep the vector, keep the tombstone.
                    None => report.deferred += 1,
                    Some(digest) => {
                        let n = del
                            .execute(rusqlite::params![node_id, digest])
                            .context("delete stale vector")?;
                        if n > 0 {
                            report.stale += n;
                        } else {
                            report.unchanged += 1;
                        }
                        resolved.push(*node_id);
                    }
                }
            }
        }

        // ── 3. doc-space sweep ───────────────────────────────────────────────
        if verify_docs {
            let doc_rows: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM edb.node_embeddings e \
                     JOIN nodes n ON n.id = e.node_id WHERE n.kind = 'doc-chunk'",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            if doc_rows > DOC_VERIFY_MAX_ROWS {
                tracing::warn!(
                    doc_rows,
                    cap = DOC_VERIFY_MAX_ROWS,
                    "doc-space too large for full verification — falling back to CDC-only"
                );
            } else {
                // Digest in the row mapper, not after collecting: `DOC_VERIFY_MAX_ROWS`
                // bounds this sweep in *rows*, but a doc chunk carries up to the
                // `EmbedRichness::Standard` cap of prose, so collecting the text
                // would let a corpus at the cap hold ~500 MB at once inside a write
                // transaction. Digesting per row makes the cap bound bytes too.
                // The rows must still be collected rather than streamed: the
                // SELECT's EXISTS subquery reads the same table the loop below
                // deletes from, and SQLite leaves that interleaving undefined.
                let docs: Vec<(i64, String)> = {
                    let mut stmt = conn
                        .prepare(
                            "SELECT n.id, n.embed_text FROM nodes n \
                             WHERE n.kind = 'doc-chunk' AND n.embed_text IS NOT NULL \
                             AND EXISTS (SELECT 1 FROM edb.node_embeddings e \
                                         WHERE e.node_id = n.id)",
                        )
                        .context("prepare doc verification query")?;
                    let rows = stmt
                        .query_map([], |r| {
                            Ok((r.get::<_, i64>(0)?, text_hash(&r.get::<_, String>(1)?)))
                        })
                        .context("run doc verification query")?;
                    rows.collect::<rusqlite::Result<Vec<_>>>()
                        .context("decode doc verification rows")?
                };
                let mut del = conn
                    .prepare(
                        "DELETE FROM edb.node_embeddings \
                         WHERE node_id = ?1 AND (text_hash IS NULL OR text_hash <> ?2)",
                    )
                    .context("prepare doc stale-vector delete")?;
                for (node_id, digest) in &docs {
                    report.doc_stale += del
                        .execute(rusqlite::params![node_id, digest])
                        .context("delete stale doc vector")?;
                }
            }
        }

        // ── 4. ack ───────────────────────────────────────────────────────────
        if ack {
            // Orphan tombstones (node row already gone) are resolved by step 1.
            conn.execute(
                "DELETE FROM node_tombstones \
                 WHERE node_id NOT IN (SELECT id FROM nodes)",
                [],
            )
            .context("ack orphan tombstones")?;
            let mut ack_stmt = conn
                .prepare("DELETE FROM node_tombstones WHERE node_id = ?1")
                .context("prepare tombstone ack")?;
            for node_id in &resolved {
                ack_stmt
                    .execute([node_id])
                    .context("ack verified tombstone")?;
            }
        }
        Ok(())
    })();

    match result {
        Ok(()) => {
            conn.execute_batch("COMMIT")
                .context("commit invalidation transaction")?;
            if report != InvalidationReport::default() {
                tracing::info!(
                    unchanged = report.unchanged,
                    stale = report.stale,
                    orphaned = report.orphaned,
                    deferred = report.deferred,
                    not_embedded = report.not_embedded,
                    doc_stale = report.doc_stale,
                    "content-hash invalidation pass"
                );
            }
            Ok(report)
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Self-cleaning scratch dir — this crate has no `tempfile` dev-dependency
    /// (see index.rs's tests, same pattern), and the audited dependency set is
    /// not worth growing for a test fixture.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static SEQ: AtomicUsize = AtomicUsize::new(0);
            let p = std::env::temp_dir().join(format!(
                "travsr_embed_{tag}_{}_{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Build a graph.db + attached embed.db with the v17 tombstone schema.
    fn setup() -> (TempDir, Connection) {
        let tmp = TempDir::new("fresh");
        let graph = tmp.path().join("graph.db");
        let embed = tmp.path().join("embed.db");
        let conn = Connection::open(&graph).unwrap();
        conn.execute_batch(
            "CREATE TABLE nodes (
                 id INTEGER PRIMARY KEY, corpus TEXT, root TEXT, path TEXT,
                 language TEXT, signature TEXT, kind TEXT, package TEXT,
                 line INTEGER, end_line INTEGER, shell_number INTEGER,
                 embed_text TEXT);
             CREATE TABLE node_tombstones (
                 node_id INTEGER NOT NULL,
                 deleted_at INTEGER NOT NULL DEFAULT (unixepoch()));
             CREATE TRIGGER capture_node_delete AFTER DELETE ON nodes
             BEGIN INSERT INTO node_tombstones(node_id) VALUES (OLD.id); END;",
        )
        .unwrap();
        conn.execute_batch(&format!(
            "ATTACH DATABASE '{}' AS edb",
            embed.to_str().unwrap()
        ))
        .unwrap();
        ensure_schema(&conn, "edb.").unwrap();
        (tmp, conn)
    }

    fn add_node(conn: &Connection, id: i64, kind: &str, text: Option<&str>) {
        conn.execute(
            "INSERT INTO nodes (id, corpus, root, path, language, signature, kind, package, embed_text) \
             VALUES (?1, 'c', '', 'docs/a.md', 'markdown', ?2, ?3, '', ?4)",
            rusqlite::params![id, format!("sig{id}"), kind, text],
        )
        .unwrap();
    }

    fn add_embedding(conn: &Connection, id: i64, hash: Option<String>) {
        conn.execute(
            "INSERT INTO edb.node_embeddings (node_id, model_id, embedding, text_hash) \
             VALUES (?1, 'm', X'00', ?2)",
            rusqlite::params![id, hash],
        )
        .unwrap();
    }

    fn embedding_ids(conn: &Connection) -> Vec<i64> {
        let mut stmt = conn
            .prepare("SELECT node_id FROM edb.node_embeddings ORDER BY node_id")
            .unwrap();
        let v: Vec<i64> = stmt
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        v
    }

    #[test]
    fn hash_is_stable_and_content_sensitive() {
        assert_eq!(text_hash("alpha"), text_hash("alpha"));
        assert_ne!(text_hash("alpha"), text_hash("alphb"));
        assert_ne!(text_hash("alpha"), text_hash("alpha "));
        assert_eq!(text_hash("").len(), 32);
    }

    /// W3's gate: re-indexing a file tombstones every chunk in it, but only the
    /// chunk whose text actually changed may lose its vector.
    #[test]
    fn reindex_churn_keeps_unchanged_chunks() {
        let (_tmp, conn) = setup();
        for id in 1..=3 {
            let text = format!("chunk {id} body");
            add_node(&conn, id, "doc-chunk", Some(&text));
            add_embedding(&conn, id, Some(text_hash(&text)));
        }
        // Simulate reindex_replace: delete all three (trigger writes tombstones),
        // re-insert with only chunk 2's body edited.
        conn.execute("DELETE FROM nodes", []).unwrap();
        for id in 1..=3 {
            let text = if id == 2 {
                "chunk 2 body EDITED".to_string()
            } else {
                format!("chunk {id} body")
            };
            add_node(&conn, id, "doc-chunk", Some(&text));
        }

        let report = apply_invalidation(&conn, true, true).unwrap();
        assert_eq!(report.stale, 1, "only the edited chunk is invalidated");
        assert_eq!(report.unchanged, 2);
        assert_eq!(embedding_ids(&conn), vec![1, 3]);
        let pending: i64 = conn
            .query_row("SELECT COUNT(*) FROM node_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(pending, 0, "verified tombstones are acked");
    }

    /// A deleted file's vectors must leave embed.db even though its tombstones
    /// and its node rows are both gone by the time a pass runs.
    #[test]
    fn deleted_nodes_lose_their_vectors_even_without_tombstones() {
        let (_tmp, conn) = setup();
        add_node(&conn, 1, "doc-chunk", Some("kept"));
        add_embedding(&conn, 1, Some(text_hash("kept")));
        add_embedding(&conn, 99, Some("whatever".into())); // node row never existed

        let report = apply_invalidation(&conn, true, true).unwrap();
        assert_eq!(report.orphaned, 1);
        assert_eq!(embedding_ids(&conn), vec![1]);
    }

    /// F3: a node whose `embed_text` is NULL (fresh `travsr init`) must not be
    /// invalidated — deleting here would re-embed it from the heading-only
    /// fallback. The tombstone is kept so a later pass re-checks it.
    #[test]
    fn null_embed_text_defers_instead_of_invalidating() {
        let (_tmp, conn) = setup();
        add_node(&conn, 1, "doc-chunk", Some("body"));
        add_embedding(&conn, 1, Some(text_hash("body")));
        conn.execute("DELETE FROM nodes", []).unwrap();
        add_node(&conn, 1, "doc-chunk", None);

        let report = apply_invalidation(&conn, true, true).unwrap();
        assert_eq!(report.deferred, 1);
        assert_eq!(report.stale, 0);
        assert_eq!(embedding_ids(&conn), vec![1], "vector survives");
        let pending: i64 = conn
            .query_row("SELECT COUNT(*) FROM node_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(pending, 1, "deferred tombstone is not acked");
    }

    /// The doc sweep repairs vectors nothing ever tombstoned: a legacy row with
    /// no hash, and a row whose text changed while its tombstone was lost.
    #[test]
    fn doc_sweep_repairs_untombstoned_rows() {
        let (_tmp, conn) = setup();
        add_node(&conn, 1, "doc-chunk", Some("current"));
        add_embedding(&conn, 1, None); // legacy / degraded: unverifiable
        add_node(&conn, 2, "doc-chunk", Some("current"));
        add_embedding(&conn, 2, Some(text_hash("stale text")));
        add_node(&conn, 3, "doc-chunk", Some("current"));
        add_embedding(&conn, 3, Some(text_hash("current")));
        // A code node with an unverifiable row must NOT be swept: the code space
        // is CDC-driven only.
        add_node(&conn, 4, "function", Some("fn body"));
        add_embedding(&conn, 4, None);

        let report = apply_invalidation(&conn, true, true).unwrap();
        assert_eq!(report.doc_stale, 2);
        assert_eq!(embedding_ids(&conn), vec![3, 4]);
        assert!(report.removed_any());
    }

    /// Node ids the next embedding pass would pick up, using the **same two
    /// conditions** the real pending-node query in `main.rs` applies:
    /// [`crate::index::NODE_ELIGIBLE`] and "has no row in `node_embeddings` for
    /// this model". Written against the shared `NODE_ELIGIBLE` constant rather
    /// than a copy of the predicate, so a change to embedding eligibility that
    /// broke recovery would fail this test instead of silently passing it.
    fn would_be_reembedded(conn: &Connection, model_id: &str) -> Vec<i64> {
        let sql = format!(
            "SELECT n.id FROM nodes n \
             WHERE {} \
             AND NOT EXISTS (SELECT 1 FROM edb.node_embeddings e \
                             WHERE e.node_id = n.id AND e.model_id = ?1) \
             ORDER BY n.id",
            crate::index::NODE_ELIGIBLE
        );
        let mut stmt = conn.prepare(&sql).unwrap();
        stmt.query_map([model_id], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    }

    /// **G4 / O4: the kill-between-ack-and-embed recovery path**, which #376
    /// §19.3 recorded as reasoned but never exercised.
    ///
    /// An invalidation pass acks tombstones and deletes stale vectors in one
    /// committed transaction, and only *then* starts embedding. A crash, a
    /// SIGKILL, or the no-progress watchdog firing in that window leaves the
    /// affected nodes with **no vector and no tombstone** — the tombstone, which
    /// is the record that anything was wrong, is already gone. Nothing would
    /// ever repair them if recovery depended on tombstones.
    ///
    /// It does not: recovery rides on the ordinary coverage query, which selects
    /// on the *absence of a vector*. This test simulates the kill by doing
    /// exactly what the pass does and then stopping — no embedding — and asserts
    /// the affected nodes are queued for the next pass anyway.
    #[test]
    fn a_kill_between_ack_and_embed_leaves_every_affected_node_queued() {
        let (_tmp, conn) = setup();
        // 1 and 2 have edited prose (their vectors are stale), 3 is untouched.
        for id in 1..=3 {
            let text = format!("chunk {id} body");
            add_node(&conn, id, "doc-chunk", Some(&text));
            add_embedding(&conn, id, Some(text_hash(&text)));
        }
        conn.execute("DELETE FROM nodes", []).unwrap();
        for id in 1..=3 {
            let text = if id == 3 {
                format!("chunk {id} body")
            } else {
                format!("chunk {id} body EDITED")
            };
            add_node(&conn, id, "doc-chunk", Some(&text));
        }

        // The pass runs to completion and commits: vectors deleted, tombstones
        // acked. Then the process dies here, before a single row is embedded.
        let report = apply_invalidation(&conn, true, true).unwrap();
        assert_eq!(report.stale, 2);
        assert_eq!(embedding_ids(&conn), vec![3]);

        let orphaned_tombstones: i64 = conn
            .query_row("SELECT COUNT(*) FROM node_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            orphaned_tombstones, 0,
            "precondition: the ack already happened, so no tombstone remains to \
             drive recovery — this is exactly what makes the window dangerous"
        );

        // The next pass must pick both nodes up with no manual command.
        assert_eq!(
            would_be_reembedded(&conn, "m"),
            vec![1, 2],
            "every node whose vector the killed pass deleted must be queued for \
             the next one, on vector-absence alone"
        );
        assert!(
            !would_be_reembedded(&conn, "m").contains(&3),
            "an untouched node must not be re-embedded — recovery repairs the \
             damage, it does not redo the repo"
        );
    }

    /// The recovery above must not resurrect the F3 hazard: a node whose
    /// `embed_text` is NULL keeps its vector *and* its tombstone, so a kill in
    /// the same window leaves it out of the queue rather than queued to be
    /// re-embedded from the heading-only fallback.
    #[test]
    fn kill_recovery_does_not_queue_null_embed_text_nodes() {
        let (_tmp, conn) = setup();
        add_node(&conn, 1, "doc-chunk", Some("body"));
        add_embedding(&conn, 1, Some(text_hash("body")));
        conn.execute("DELETE FROM nodes", []).unwrap();
        add_node(&conn, 1, "doc-chunk", None);

        apply_invalidation(&conn, true, true).unwrap();
        assert!(
            would_be_reembedded(&conn, "m").is_empty(),
            "a NULL-embed_text doc chunk is ineligible, so it is never queued \
             for a fallback-text embedding"
        );
    }

    /// Every tombstone on this repo at the time of writing (374 of 374) was for
    /// a node kind that is never embedded. Deferring those would leave the
    /// backlog permanently non-zero, which in turn makes the daemon's
    /// "invalidation pending" signal useless.
    #[test]
    fn tombstones_for_never_embedded_nodes_are_acked_not_deferred() {
        let (_tmp, conn) = setup();
        add_node(&conn, 1, "import", None);
        conn.execute("DELETE FROM nodes", []).unwrap();
        add_node(&conn, 1, "import", None);

        let report = apply_invalidation(&conn, true, true).unwrap();
        assert_eq!(report.not_embedded, 1);
        assert_eq!(report.deferred, 0);
        let pending: i64 = conn
            .query_row("SELECT COUNT(*) FROM node_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(pending, 0, "backlog must be able to reach zero");
    }

    #[test]
    fn no_work_reports_nothing_removed() {
        let (_tmp, conn) = setup();
        add_node(&conn, 1, "doc-chunk", Some("body"));
        add_embedding(&conn, 1, Some(text_hash("body")));
        let report = apply_invalidation(&conn, true, true).unwrap();
        assert!(!report.removed_any());
        assert_eq!(embedding_ids(&conn), vec![1]);
    }

    /// `--rebuild-index` may delete a stale vector but cannot re-embed it, so it
    /// must leave the tombstone for the next real pass.
    #[test]
    fn no_ack_keeps_tombstones_for_the_next_pass() {
        let (_tmp, conn) = setup();
        add_node(&conn, 1, "doc-chunk", Some("body"));
        add_embedding(&conn, 1, Some(text_hash("body")));
        conn.execute("DELETE FROM nodes", []).unwrap();
        add_node(&conn, 1, "doc-chunk", Some("edited body"));

        let report = apply_invalidation(&conn, false, true).unwrap();
        assert_eq!(report.stale, 1);
        let pending: i64 = conn
            .query_row("SELECT COUNT(*) FROM node_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(pending, 1);
    }

    #[test]
    fn schema_migration_adds_hash_column_to_legacy_table() {
        let tmp = TempDir::new("migrate");
        let conn = Connection::open(tmp.path().join("embed.db")).unwrap();
        // Pre-#376 table shape.
        conn.execute_batch(
            "CREATE TABLE node_embeddings (
                 node_id INTEGER NOT NULL, model_id TEXT NOT NULL,
                 embedding BLOB NOT NULL, PRIMARY KEY (node_id, model_id)
             ) WITHOUT ROWID;",
        )
        .unwrap();
        conn.execute("INSERT INTO node_embeddings VALUES (1, 'm', X'00')", [])
            .unwrap();

        ensure_schema(&conn, "").unwrap();
        let hash: Option<String> = conn
            .query_row("SELECT text_hash FROM node_embeddings", [], |r| r.get(0))
            .unwrap();
        assert!(hash.is_none(), "legacy rows migrate with a NULL hash");
        // Idempotent.
        ensure_schema(&conn, "").unwrap();
    }
}
