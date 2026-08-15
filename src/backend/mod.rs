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

    /// True when this backend must be fed from a single submit loop rather than
    /// N worker threads.
    ///
    /// Accelerated backends want this because N CPU threads do not map onto one
    /// GPU. ORT wants it even on CPU for a different reason: `Session::run`
    /// takes `&mut self`, so every submission serialises on one mutex anyway
    /// while ORT's own intra-op pool does the parallelising. Spawning workers
    /// that immediately queue behind that lock buys nothing and costs a thread
    /// each, so `OrtBackend` overrides this to true regardless of EP.
    fn single_submitter(&self) -> bool {
        self.is_accelerated()
    }
}

/// Preference tiers: accelerated > preferred-CPU (tract) > universal-CPU (ort).
/// ort/CPU is only *chosen* when it is the only engine that can run the model.
pub const PREF_ACCELERATED: i32 = 100; // also the `accelerated_compiled` threshold
pub const PREF_PREFERRED_CPU: i32 = 50;
#[cfg_attr(not(feature = "ort"), allow(dead_code))] // consumed by the ORT factory
pub const PREF_UNIVERSAL_CPU: i32 = 10;

/// Contributes one engine configuration to the registry.
pub trait BackendFactory: Send + Sync {
    /// Engine label for logs and error messages, e.g. "tract", "ort-accelerated".
    fn engine(&self) -> &'static str;
    /// Capability: can this engine execute the given model architecture family?
    fn can_run(&self, family: &str) -> bool;
    /// The families `can_run` accepts, or `None` when the engine is universal
    /// (accepts any family). This is `can_run` turned inside out so the
    /// `--capabilities` handshake can *enumerate* what this build supports —
    /// `can_run` alone can only answer one family at a time, which is no use to
    /// a CLI deciding whether a model is selectable before it downloads it.
    fn supported_families(&self) -> Option<&'static [&'static str]>;
    /// Ordering among capable engines.
    fn preference(&self) -> i32;
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
    #[cfg(any(
        feature = "ort-coreml",
        feature = "ort-cuda",
        feature = "ort-directml",
        feature = "ort-tensorrt",
        feature = "ort-webgpu"
    ))]
    factories.push(Box::new(ort::OrtFactory::accelerated()));
    factories.push(Box::new(tract::TractFactory));
    #[cfg(feature = "ort")]
    factories.push(Box::new(ort::OrtFactory::cpu()));
    factories
}

/// What this compiled sidecar can run, for the CLI's pre-flight handshake.
///
/// Reported from the registry, so it cannot drift from what the resolver will
/// actually do. Note the deliberate limit of `accelerated_compiled`: it says a
/// hardware-EP factory is *compiled in*, not that an accelerator will be
/// confirmed at run time — that answer requires loading a model and doing a
/// warm-up inference, which this flag-only path must not do.
#[derive(serde::Serialize, Debug)]
pub struct Capabilities {
    pub plugin_version: &'static str,
    /// Bumped when the shape of this payload changes, so a newer CLI can tell
    /// an older sidecar's format from its own.
    pub capabilities_version: u32,
    pub engines: Vec<&'static str>,
    /// Families this build can run *by name*. Not exhaustive when
    /// `universal_onnx` is true — see that field.
    pub families: Vec<&'static str>,
    /// True when a universal ONNX engine is compiled in, meaning families
    /// outside `families` are very likely runnable too. A CLI must treat
    /// `families` as a hard allowlist only when this is false.
    pub universal_onnx: bool,
    pub accelerated_compiled: bool,
}

pub fn capabilities() -> Capabilities {
    let factories = registry();
    let mut families: Vec<&'static str> = Vec::new();
    let mut universal_onnx = false;
    for f in &factories {
        match f.supported_families() {
            None => universal_onnx = true,
            Some(list) => {
                for fam in list {
                    if !families.contains(fam) {
                        families.push(fam);
                    }
                }
            }
        }
    }
    Capabilities {
        plugin_version: env!("CARGO_PKG_VERSION"),
        capabilities_version: 1,
        engines: factories.iter().map(|f| f.engine()).collect(),
        families,
        universal_onnx,
        accelerated_compiled: factories.iter().any(|f| f.preference() >= PREF_ACCELERATED),
    }
}

/// Resolve and build the backend for this model on this machine.
/// Logs the chosen backend + execution provider at startup (always).
pub fn create_backend(model_dir: &Path, desc: &ModelDescriptor) -> Result<Arc<dyn EmbedBackend>> {
    resolve(registry(), model_dir, desc)
}

fn resolve(
    mut factories: Vec<Box<dyn BackendFactory>>,
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
    factories.sort_by_key(|f| Reverse(f.preference()));

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
        fn supported_families(&self) -> Option<&'static [&'static str]> {
            self.families
        }
        fn preference(&self) -> i32 {
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

    fn run(factories: Vec<Box<dyn BackendFactory>>, family: &str) -> Result<Arc<dyn EmbedBackend>> {
        resolve(factories, Path::new("/nonexistent"), &desc(family))
    }

    fn run_expecting_err(factories: Vec<Box<dyn BackendFactory>>, family: &str) -> String {
        match run(factories, family) {
            Ok(b) => panic!("expected error, got backend '{}'", b.backend_name()),
            Err(e) => format!("{e:#}"),
        }
    }

    /// Root directory holding the model fixtures the `#[ignore]` tests below need.
    ///
    /// `TRAVSR_EMBED_TEST_MODEL_DIR` overrides it so CI can stage fixtures
    /// somewhere cacheable (see .github/scripts/fetch-test-models.sh) instead of
    /// requiring a real user install under $HOME.
    #[cfg_attr(not(feature = "ort"), allow(dead_code))]
    fn test_model_root() -> std::path::PathBuf {
        match std::env::var_os("TRAVSR_EMBED_TEST_MODEL_DIR") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => dirs::home_dir()
                .expect("neither TRAVSR_EMBED_TEST_MODEL_DIR nor HOME is set")
                .join(".travsr/models"),
        }
    }

    #[cfg_attr(not(feature = "ort"), allow(dead_code))]
    fn fixture(name: &str) -> std::path::PathBuf {
        let dir = test_model_root().join(name);
        assert!(
            dir.join("model.onnx").exists(),
            "fixture '{name}' not installed at {} — run .github/scripts/fetch-test-models.sh \
             or set TRAVSR_EMBED_TEST_MODEL_DIR",
            dir.display()
        );
        dir
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
        let msg = match resolve(registry(), Path::new("/nonexistent"), &desc("modernbert")) {
            Ok(b) => panic!("expected error, got backend '{}'", b.backend_name()),
            Err(e) => format!("{e:#}"),
        };
        assert!(msg.contains("modernbert") && msg.contains("ORT"), "{msg}");
    }

    /// tract's allowlist has to reflect a real limitation, not a guess. This
    /// bypasses the allowlist and hands tract a genuine ModernBERT graph (RoPE +
    /// alternating local/global attention + GeGLU): it must FAIL to load.
    ///
    /// If this ever starts passing, tract gained ModernBERT support and
    /// `TRACT_FAMILIES` should grow — the test failing is the signal to look.
    #[test]
    #[ignore = "needs the tiny-modernbert fixture (see CONTRIBUTING.md)"]
    fn tract_refuses_a_real_modernbert_graph() {
        let dir = fixture("tiny-modernbert");
        let md = ModelDescriptor::load(&dir).expect("fixture model.toml");
        assert_eq!(md.family, "modernbert", "fixture must be tagged modernbert");
        let err = tract::TractBackend::load(&dir, md)
            .err()
            .expect("tract must not be able to load a ModernBERT graph");
        tracing::info!("tract rejected ModernBERT as expected: {err:#}");
    }

    /// The extensibility claim from issue #6, end to end on a real graph: a
    /// ModernBERT model runs with NO sidecar code beyond `family = "modernbert"`
    /// in its model.toml. tract is capability-filtered out; ORT runs it and
    /// produces usable vectors.
    #[cfg(feature = "ort")]
    #[test]
    #[ignore = "needs the tiny-modernbert fixture (see CONTRIBUTING.md)"]
    fn ort_runs_a_real_modernbert_graph_with_only_a_family_tag() {
        let dir = fixture("tiny-modernbert");
        let md = ModelDescriptor::load(&dir).expect("fixture model.toml");
        let backend = resolve(registry(), &dir, &md).expect("ORT must run the modernbert family");
        assert!(
            backend.backend_name().starts_with("ort/"),
            "expected an ORT backend, got '{}'",
            backend.backend_name()
        );

        let blobs = backend
            .embed_documents(&["fn parse_manifest", "class PodController"])
            .expect("embed with ORT");
        assert_eq!(blobs.len(), 2);
        for blob in &blobs {
            let v = crate::model::blob_to_f32(blob);
            assert_eq!(v.len(), md.output_dim());
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "vectors must be l2-normalized, got norm {norm}"
            );
        }
    }

    /// CoreML on Apple Silicon: the accelerated factory must actually CONFIRM the
    /// EP (not decline), and its vectors must match tract's within the issue #6
    /// drift bound. Runs for real on the macos-14 (M1) CI runner.
    #[cfg(feature = "ort-coreml")]
    #[test]
    #[ignore = "needs Apple Silicon + the bge-small-en-v1.5 fixture"]
    fn coreml_confirms_and_matches_tract() {
        let dir = fixture("bge-small-en-v1.5");
        let md = ModelDescriptor::load(&dir).unwrap();

        let ort_backend = ort::OrtFactory::accelerated()
            .try_build(&dir, &md)
            .expect("CoreML factory must not error")
            .expect("CoreML must be CONFIRMED on Apple Silicon, not declined");
        assert_eq!(ort_backend.backend_name(), "ort/CoreML");
        assert!(ort_backend.is_accelerated());

        let tract_backend = tract::TractFactory.try_build(&dir, &md).unwrap().unwrap();
        let texts = ["function: parseManifest", "class: PodController"];
        let a = tract_backend.embed_documents(&texts).unwrap();
        let b = ort_backend.embed_documents(&texts).unwrap();
        for (blob_a, blob_b) in a.iter().zip(&b) {
            let va = crate::model::blob_to_f32(blob_a);
            let vb = crate::model::blob_to_f32(blob_b);
            let cos: f32 = va.iter().zip(&vb).map(|(x, y)| x * y).sum();
            // Looser than the ort/CPU bound: CoreML may run fp16 on the ANE.
            assert!(cos >= 0.99, "tract vs CoreML cosine too low: {cos}");
        }
    }

    // ── --capabilities handshake ─────────────────────────────────────────────

    /// True for every build: the payload must describe the registry it was
    /// built from, whatever features are on.
    #[test]
    fn capabilities_reports_the_real_registry() {
        let c = capabilities();
        assert_eq!(c.capabilities_version, 1);
        assert_eq!(c.plugin_version, env!("CARGO_PKG_VERSION"));
        let engines: Vec<&str> = registry().iter().map(|f| f.engine()).collect();
        assert_eq!(c.engines, engines);
        // tract is always compiled in, so its families are always enumerable.
        assert!(c.families.contains(&"bert"), "{:?}", c.families);
        assert!(c.families.contains(&"minilm"), "{:?}", c.families);
        // Deduped: two factories reporting overlapping lists must not double up.
        let mut sorted = c.families.clone();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len(), "families contains duplicates");
    }

    /// The default build is the one the travsr CLI must treat `families` as a
    /// hard allowlist for — no universal engine means a family outside the list
    /// genuinely cannot run.
    #[cfg(not(feature = "ort"))]
    #[test]
    fn capabilities_on_default_build_is_a_closed_allowlist() {
        let c = capabilities();
        assert_eq!(c.engines, vec!["tract"]);
        assert!(!c.universal_onnx);
        assert!(!c.accelerated_compiled);
    }

    #[cfg(feature = "ort")]
    #[test]
    fn capabilities_with_ort_is_universal() {
        let c = capabilities();
        assert!(
            c.universal_onnx,
            "an ORT build must advertise universal ONNX support"
        );
    }

    /// Compiling a hardware EP is what `accelerated_compiled` claims — nothing
    /// about whether this machine has the hardware.
    #[cfg(any(
        feature = "ort-coreml",
        feature = "ort-cuda",
        feature = "ort-directml",
        feature = "ort-tensorrt",
        feature = "ort-webgpu"
    ))]
    #[test]
    fn capabilities_with_hw_ep_feature_reports_accelerated_compiled() {
        assert!(capabilities().accelerated_compiled);
    }

    #[test]
    fn capabilities_serializes_to_json_with_the_documented_keys() {
        let json = serde_json::to_string(&capabilities()).unwrap();
        for key in [
            "plugin_version",
            "capabilities_version",
            "engines",
            "families",
            "universal_onnx",
            "accelerated_compiled",
        ] {
            assert!(json.contains(key), "missing key '{key}' in {json}");
        }
    }

    // ── single_submitter ────────────────────────────────────────────────────

    /// The trait default ties single-submit to acceleration; `OrtBackend`
    /// overrides it (see its impl). Both halves matter, so pin the default here.
    #[test]
    fn single_submitter_defaults_to_is_accelerated() {
        let cpu = FakeBackend {
            name: "x",
            accelerated: false,
        };
        let gpu = FakeBackend {
            name: "y",
            accelerated: true,
        };
        assert!(!cpu.single_submitter());
        assert!(gpu.single_submitter());
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
    /// Needs the bge-small-en-v1.5 fixture on disk — run explicitly:
    ///   cargo test --features ort -- --ignored
    #[cfg(feature = "ort")]
    #[test]
    #[ignore = "needs the bge-small-en-v1.5 fixture (see CONTRIBUTING.md)"]
    fn tract_vs_ort_cpu_cosine_parity() {
        let dir = fixture("bge-small-en-v1.5");
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
