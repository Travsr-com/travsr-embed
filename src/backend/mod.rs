// Engine-agnostic backend layer (issue #6).
//
// The sidecar holds `Arc<dyn EmbedBackend>`; no call site names a concrete
// engine. Engines are contributed by `BackendFactory` impls, and selection is
// a CAPABILITY MATCH, not a fixed fallback chain: tract is not a slower
// fallback for ModernBERT — it cannot run it at all, so it is filtered out of
// the candidate list before preference ordering even happens.
//
// Resolver:
//   1. retain factories whose `can_run(family)` is true      (capability)
//   2. sort by `preference(target)` descending                (ordering)
//   3. first `try_build` that returns Ok(Some) wins
//      Ok(None) = declined (e.g. no confirmed HW accelerator), Err = failed
//
// The registry is assembled from compiled features — the ONLY place engines
// are listed. A new engine is a new factory impl + a `dep:` + a feature + one
// `push` here; the trait, resolver, encode pipeline, and call sites are
// untouched.

pub mod tract;

#[cfg(feature = "ort")]
pub mod ort;

use std::cmp::Reverse;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::model::ModelDescriptor;

/// A ready-to-use inference engine bound to one loaded model.
pub trait EmbedBackend: Send + Sync {
    /// Embed texts for indexing (no prefix). Returns one `dim`×4-byte BLOB per input.
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<u8>>>;
    /// Embed a single query (with the descriptor's query prefix). Returns one BLOB.
    fn embed_query(&self, text: &str) -> Result<Vec<u8>>;
    /// Provenance label: "tract", "ort/CUDA", "ort/CoreML", "ort/CPU".
    fn backend_name(&self) -> &str;
    /// True when a hardware execution provider (GPU/ANE) was confirmed.
    fn is_accelerated(&self) -> bool;
    /// Stored/output embedding dimension (native, or Matryoshka-truncated).
    fn dim(&self) -> usize;
}

/// Host characteristics available to `preference()` ordering. Today's factories
/// rank by tier alone; the fields are the extension surface for target-aware
/// ordering (e.g. preferring CoreML only on macOS/aarch64).
pub struct TargetInfo {
    #[allow(dead_code)]
    pub os: &'static str,
    #[allow(dead_code)]
    pub arch: &'static str,
}

impl TargetInfo {
    pub fn current() -> Self {
        Self {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        }
    }
}

/// Preference tiers: accelerated > preferred-CPU (tract) > universal-CPU (ort).
/// ort/CPU is only *chosen* when it is the only engine that can run the model.
#[cfg_attr(not(feature = "ort"), allow(dead_code))] // consumed by the ORT factory
pub const PREF_ACCELERATED: i32 = 100;
pub const PREF_PREFERRED_CPU: i32 = 50;
#[cfg_attr(not(feature = "ort"), allow(dead_code))] // consumed by the ORT factory
pub const PREF_UNIVERSAL_CPU: i32 = 10;

/// Contributes one engine configuration to the registry.
pub trait BackendFactory: Send + Sync {
    /// Engine label for logs and error messages, e.g. "tract", "ort-accelerated".
    fn engine(&self) -> &'static str;
    /// Capability: can this engine execute the given model architecture family?
    fn can_run(&self, family: &str) -> bool;
    /// Ordering among capable engines on this target.
    fn preference(&self, target: &TargetInfo) -> i32;
    /// Ok(Some) = chosen, Ok(None) = declined (not an error — e.g. no confirmed
    /// hardware accelerator), Err = failed (logged; resolver moves on).
    fn try_build(
        &self,
        model_dir: &Path,
        desc: &ModelDescriptor,
    ) -> Result<Option<Arc<dyn EmbedBackend>>>;
}

/// The compiled-in engine registry — the only place engines are listed.
#[allow(clippy::vec_init_then_push)] // entries are cfg-gated; vec![] can't express that
fn registry() -> Vec<Box<dyn BackendFactory>> {
    let mut factories: Vec<Box<dyn BackendFactory>> = Vec::new();
    #[cfg(any(feature = "ort-coreml", feature = "ort-cuda", feature = "ort-tensorrt"))]
    factories.push(Box::new(ort::OrtFactory::accelerated()));
    factories.push(Box::new(tract::TractFactory));
    #[cfg(feature = "ort")]
    factories.push(Box::new(ort::OrtFactory::cpu()));
    factories
}

/// Resolve and build the backend for this model on this machine.
/// Logs the chosen backend + execution provider at startup (always).
pub fn create_backend(model_dir: &Path, desc: &ModelDescriptor) -> Result<Arc<dyn EmbedBackend>> {
    resolve(registry(), &TargetInfo::current(), model_dir, desc)
}

fn resolve(
    mut factories: Vec<Box<dyn BackendFactory>>,
    target: &TargetInfo,
    model_dir: &Path,
    desc: &ModelDescriptor,
) -> Result<Arc<dyn EmbedBackend>> {
    let compiled: Vec<&str> = factories.iter().map(|f| f.engine()).collect();
    factories.retain(|f| f.can_run(&desc.family));
    anyhow::ensure!(
        !factories.is_empty(),
        "no compiled engine can run model family '{}' (compiled engines: {}) — \
         this model needs an ORT-enabled travsr-embed build",
        desc.family,
        compiled.join(", ")
    );
    factories.sort_by_key(|f| Reverse(f.preference(target)));

    let mut failures: Vec<String> = Vec::new();
    for factory in &factories {
        match factory.try_build(model_dir, desc) {
            Ok(Some(backend)) => {
                tracing::info!(
                    backend = backend.backend_name(),
                    accelerated = backend.is_accelerated(),
                    family = %desc.family,
                    dim = backend.dim(),
                    "embed backend selected"
                );
                return Ok(backend);
            }
            Ok(None) => {
                tracing::info!(engine = factory.engine(), "backend declined (not chosen)");
            }
            Err(e) => {
                tracing::warn!(engine = factory.engine(), "backend failed to build: {e:#}");
                failures.push(format!("{}: {e:#}", factory.engine()));
            }
        }
    }
    anyhow::bail!(
        "no engine could run model family '{}'{}",
        desc.family,
        if failures.is_empty() {
            String::new()
        } else {
            format!("; failures: {}", failures.join("; "))
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Pooling;

    fn desc(family: &str) -> ModelDescriptor {
        ModelDescriptor {
            dim: 4,
            pooling: Pooling::Cls,
            query_prefix: String::new(),
            n_inputs: 3,
            truncate_dim: 0,
            family: family.to_owned(),
        }
    }

    struct FakeBackend {
        name: &'static str,
        accelerated: bool,
    }

    impl EmbedBackend for FakeBackend {
        fn embed_documents(&self, _texts: &[&str]) -> Result<Vec<Vec<u8>>> {
            Ok(vec![])
        }
        fn embed_query(&self, _text: &str) -> Result<Vec<u8>> {
            Ok(vec![])
        }
        fn backend_name(&self) -> &str {
            self.name
        }
        fn is_accelerated(&self) -> bool {
            self.accelerated
        }
        fn dim(&self) -> usize {
            4
        }
    }

    /// Outcome a fake factory produces when the resolver reaches it.
    enum Outcome {
        Chosen,
        Declined,
        Fails,
    }

    struct FakeFactory {
        engine: &'static str,
        families: Option<&'static [&'static str]>, // None = universal
        pref: i32,
        accelerated: bool,
        outcome: Outcome,
    }

    impl BackendFactory for FakeFactory {
        fn engine(&self) -> &'static str {
            self.engine
        }
        fn can_run(&self, family: &str) -> bool {
            match self.families {
                None => true,
                Some(list) => list.contains(&family),
            }
        }
        fn preference(&self, _t: &TargetInfo) -> i32 {
            self.pref
        }
        fn try_build(
            &self,
            _dir: &Path,
            _desc: &ModelDescriptor,
        ) -> Result<Option<Arc<dyn EmbedBackend>>> {
            match self.outcome {
                Outcome::Chosen => Ok(Some(Arc::new(FakeBackend {
                    name: self.engine,
                    accelerated: self.accelerated,
                }))),
                Outcome::Declined => Ok(None),
                Outcome::Fails => anyhow::bail!("boom"),
            }
        }
    }

    fn tract_like(outcome: Outcome) -> Box<dyn BackendFactory> {
        Box::new(FakeFactory {
            engine: "tract",
            families: Some(&["bert", "minilm"]),
            pref: PREF_PREFERRED_CPU,
            accelerated: false,
            outcome,
        })
    }

    fn ort_accel_like(outcome: Outcome) -> Box<dyn BackendFactory> {
        Box::new(FakeFactory {
            engine: "ort-accelerated",
            families: None,
            pref: PREF_ACCELERATED,
            accelerated: true,
            outcome,
        })
    }

    fn ort_cpu_like(outcome: Outcome) -> Box<dyn BackendFactory> {
        Box::new(FakeFactory {
            engine: "ort-cpu",
            families: None,
            pref: PREF_UNIVERSAL_CPU,
            accelerated: false,
            outcome,
        })
    }

    fn target() -> TargetInfo {
        TargetInfo::current()
    }

    fn run(factories: Vec<Box<dyn BackendFactory>>, family: &str) -> Result<Arc<dyn EmbedBackend>> {
        resolve(
            factories,
            &target(),
            Path::new("/nonexistent"),
            &desc(family),
        )
    }

    fn run_expecting_err(factories: Vec<Box<dyn BackendFactory>>, family: &str) -> String {
        match run(factories, family) {
            Ok(b) => panic!("expected error, got backend '{}'", b.backend_name()),
            Err(e) => format!("{e:#}"),
        }
    }

    // ── Outcome table from issue #6 ──────────────────────────────────────────

    #[test]
    fn bert_on_cpu_only_build_chooses_tract() {
        // CPU-only tract build: registry is just tract.
        let b = run(vec![tract_like(Outcome::Chosen)], "bert").unwrap();
        assert_eq!(b.backend_name(), "tract");
    }

    #[test]
    fn bert_with_confirmed_accelerator_chooses_accelerated_ort() {
        let b = run(
            vec![
                ort_accel_like(Outcome::Chosen),
                tract_like(Outcome::Chosen),
                ort_cpu_like(Outcome::Chosen),
            ],
            "bert",
        )
        .unwrap();
        assert_eq!(b.backend_name(), "ort-accelerated");
        assert!(b.is_accelerated());
    }

    #[test]
    fn bert_with_declined_accelerator_falls_back_to_tract_not_ort_cpu() {
        // Accelerated ORT declines (no HW EP confirmed) → tract is preferred
        // over the universal ort/CPU engine for a CPU-runnable model.
        let b = run(
            vec![
                ort_accel_like(Outcome::Declined),
                tract_like(Outcome::Chosen),
                ort_cpu_like(Outcome::Chosen),
            ],
            "bert",
        )
        .unwrap();
        assert_eq!(b.backend_name(), "tract");
        assert!(!b.is_accelerated());
    }

    #[test]
    fn modernbert_filters_tract_out_and_uses_accelerated_ort() {
        let b = run(
            vec![
                ort_accel_like(Outcome::Chosen),
                tract_like(Outcome::Chosen),
                ort_cpu_like(Outcome::Chosen),
            ],
            "modernbert",
        )
        .unwrap();
        assert_eq!(b.backend_name(), "ort-accelerated");
    }

    #[test]
    fn modernbert_on_cpu_only_ort_build_uses_ort_cpu() {
        // ORT-CPU is chosen only because it is the sole engine that can run
        // the family (accelerated declined, tract filtered out).
        let b = run(
            vec![
                ort_accel_like(Outcome::Declined),
                tract_like(Outcome::Chosen),
                ort_cpu_like(Outcome::Chosen),
            ],
            "modernbert",
        )
        .unwrap();
        assert_eq!(b.backend_name(), "ort-cpu");
    }

    #[test]
    fn modernbert_on_tract_only_build_errors_loudly() {
        let msg = run_expecting_err(vec![tract_like(Outcome::Chosen)], "modernbert");
        assert!(
            msg.contains("modernbert"),
            "message names the family: {msg}"
        );
        assert!(
            msg.contains("ORT"),
            "message points at the ORT build: {msg}"
        );
    }

    #[test]
    fn factory_error_moves_on_to_next_candidate() {
        let b = run(
            vec![ort_accel_like(Outcome::Fails), tract_like(Outcome::Chosen)],
            "bert",
        )
        .unwrap();
        assert_eq!(b.backend_name(), "tract");
    }

    #[test]
    fn all_candidates_fail_reports_every_failure() {
        let msg = run_expecting_err(
            vec![ort_accel_like(Outcome::Fails), ort_cpu_like(Outcome::Fails)],
            "modernbert",
        );
        assert!(
            msg.contains("ort-accelerated") && msg.contains("ort-cpu"),
            "{msg}"
        );
    }

    // ── Real registry / factory properties ──────────────────────────────────

    #[test]
    fn default_registry_contains_tract() {
        assert!(registry().iter().any(|f| f.engine() == "tract"));
    }

    /// On the default (tract-only) build, a ModernBERT model must be refused at
    /// backend resolution with an actionable message — the real registry, not fakes.
    #[cfg(not(feature = "ort"))]
    #[test]
    fn real_registry_refuses_modernbert_on_tract_only_build() {
        let msg = match resolve(
            registry(),
            &target(),
            Path::new("/nonexistent"),
            &desc("modernbert"),
        ) {
            Ok(b) => panic!("expected error, got backend '{}'", b.backend_name()),
            Err(e) => format!("{e:#}"),
        };
        assert!(msg.contains("modernbert") && msg.contains("ORT"), "{msg}");
    }

    /// With the ORT engine compiled in, the same ModernBERT descriptor resolves
    /// to ort/CPU on a machine with no accelerator — proving a new ORT-runnable
    /// architecture needs only a `family` tag, zero sidecar code. Uses the real
    /// registry but a real model file is still required, so reuse bge's ONNX
    /// (a standard BERT graph is also a valid "modernbert-tagged" stand-in for
    /// resolution purposes — the resolver never inspects the graph).
    #[cfg(feature = "ort")]
    #[test]
    #[ignore = "needs a locally installed bge-small-en-v1.5 model"]
    fn real_registry_resolves_modernbert_tag_to_ort() {
        let dir = dirs::home_dir()
            .expect("HOME not set")
            .join(".travsr/models/bge-small-en-v1.5");
        let mut md = ModelDescriptor::load(&dir).unwrap();
        md.family = "modernbert".to_owned();
        let backend =
            resolve(registry(), &target(), &dir, &md).expect("ort must run modernbert family");
        assert!(
            backend.backend_name().starts_with("ort/"),
            "{}",
            backend.backend_name()
        );
    }

    #[test]
    fn tract_allowlist_excludes_modernbert() {
        let f = tract::TractFactory;
        assert!(f.can_run("bert"));
        assert!(f.can_run("minilm"));
        assert!(!f.can_run("modernbert"));
        assert!(!f.can_run("nomic-bert"));
    }

    /// Cross-backend numerical parity: tract fp32 vs ort/CPU fp32 on the real
    /// local model. Engines differ only in matmul accumulation order, so cosine
    /// similarity between their vectors must be ≥ 0.999 (issue #6 drift bound).
    ///
    /// Needs `~/.travsr/models/bge-small-en-v1.5` on disk — run explicitly:
    ///   cargo test --features ort -- --ignored
    #[cfg(feature = "ort")]
    #[test]
    #[ignore = "needs a locally installed bge-small-en-v1.5 model"]
    fn tract_vs_ort_cpu_cosine_parity() {
        let dir = dirs::home_dir()
            .expect("HOME not set")
            .join(".travsr/models/bge-small-en-v1.5");
        assert!(
            dir.join("model.onnx").exists(),
            "model not installed: {}",
            dir.display()
        );
        let md = ModelDescriptor::load(&dir).unwrap();

        let tract_backend = tract::TractFactory
            .try_build(&dir, &md)
            .unwrap()
            .expect("tract must build for bert");
        let ort_backend = ort::OrtFactory::cpu()
            .try_build(&dir, &md)
            .unwrap()
            .expect("ort/CPU must build");
        assert_eq!(tract_backend.backend_name(), "tract");
        assert_eq!(ort_backend.backend_name(), "ort/CPU");

        let texts = [
            "function: parseManifest | module: pkg/config",
            "method: GraphTraverser.visit | callers: buildIndex, resolveConflict",
            "class: PodController",
        ];
        let a = tract_backend.embed_documents(&texts).unwrap();
        let b = ort_backend.embed_documents(&texts).unwrap();
        assert_eq!(a.len(), texts.len());
        for (blob_a, blob_b) in a.iter().zip(&b) {
            let va = crate::model::blob_to_f32(blob_a);
            let vb = crate::model::blob_to_f32(blob_b);
            assert_eq!(va.len(), md.output_dim());
            // Both vectors are l2-normalized by the shared encode pipeline, so
            // the dot product is the cosine similarity.
            let cos: f32 = va.iter().zip(&vb).map(|(x, y)| x * y).sum();
            assert!(cos >= 0.999, "cross-backend cosine parity too low: {cos}");
        }

        let qa =
            crate::model::blob_to_f32(&tract_backend.embed_query("validate user input").unwrap());
        let qb =
            crate::model::blob_to_f32(&ort_backend.embed_query("validate user input").unwrap());
        let cos: f32 = qa.iter().zip(&qb).map(|(x, y)| x * y).sum();
        assert!(cos >= 0.999, "query cosine parity too low: {cos}");
    }
}
