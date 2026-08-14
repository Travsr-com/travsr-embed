// OnnxRuntime engine (issue #6) — the universal ONNX executor.
//
// Two registry entries share this module:
//
//   OrtFactory::accelerated()  PREF_ACCELERATED — hardware EPs only. Walks the
//                              compiled EP cascade (CUDA → TensorRT → CoreML);
//                              every EP is registered with error_on_failure so
//                              ORT's silent in-session CPU fallback becomes a
//                              hard error we catch, and a warm-up inference
//                              confirms the session actually executes. If no
//                              hardware EP confirms, the factory DECLINES
//                              (Ok(None)) so the resolver can prefer tract for
//                              CPU-runnable models.
//
//   OrtFactory::cpu()          PREF_UNIVERSAL_CPU — plain CPU session. Chosen
//                              only when it is the sole engine that can run the
//                              model's family (e.g. ModernBERT on a CPU-only
//                              machine): tract is capability-filtered out and
//                              the accelerated entry declined.
//
// `can_run` is true for everything — ORT implements (near-)all of ONNX, so a
// new ORT-runnable architecture needs zero inference code here, only a
// `family` tag in model.toml.
//
// Accelerated sessions pad inputs to fixed shape buckets (encode::GPU_BUCKETS):
// CUDA/TensorRT re-optimize kernels per tensor shape, so a handful of fixed
// shapes beats per-batch BatchLongest shapes. Padded rows/columns carry
// attention-mask 0 and are dropped before pooling — output values are
// unaffected, only tensor shapes change.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use super::{BackendFactory, EmbedBackend, PREF_ACCELERATED, PREF_UNIVERSAL_CPU};
use crate::encode::{BucketSpec, Encoder, TokenBatch, GPU_BUCKETS};
use crate::model::ModelDescriptor;

/// ort's Error<S> is not Send+Sync in 2.0.0-rc.12, so `?` cannot convert it
/// into anyhow::Error — stringify at the boundary instead.
fn ort_err<S>(e: ort::Error<S>) -> anyhow::Error {
    anyhow::anyhow!("ort: {e}")
}

pub struct OrtFactory {
    accelerated: bool,
}

impl OrtFactory {
    // Registered only when a hardware EP feature is compiled in.
    #[cfg_attr(
        not(any(
            feature = "ort-coreml",
            feature = "ort-cuda",
            feature = "ort-tensorrt",
            feature = "ort-webgpu"
        )),
        allow(dead_code)
    )]
    pub fn accelerated() -> Self {
        Self { accelerated: true }
    }

    pub fn cpu() -> Self {
        Self { accelerated: false }
    }
}

impl BackendFactory for OrtFactory {
    fn engine(&self) -> &'static str {
        if self.accelerated {
            "ort-accelerated"
        } else {
            "ort-cpu"
        }
    }

    // Universal ONNX engine: every family is a candidate.
    fn can_run(&self, _family: &str) -> bool {
        true
    }

    // None = universal, matching `can_run` above. There is no finite list to
    // report, and inventing one would make the handshake claim a smaller
    // capability than the engine has.
    fn supported_families(&self) -> Option<&'static [&'static str]> {
        None
    }

    fn preference(&self) -> i32 {
        if self.accelerated {
            PREF_ACCELERATED
        } else {
            PREF_UNIVERSAL_CPU
        }
    }

    fn try_build(
        &self,
        model_dir: &Path,
        desc: &ModelDescriptor,
    ) -> Result<Option<Arc<dyn EmbedBackend>>> {
        if self.accelerated {
            OrtBackend::load_accelerated(model_dir, desc.clone())
                .map(|opt| opt.map(|b| Arc::new(b) as Arc<dyn EmbedBackend>))
        } else {
            let backend = OrtBackend::load_cpu(model_dir, desc.clone())?;
            Ok(Some(Arc::new(backend)))
        }
    }
}

/// One hardware execution provider attempt in the cascade, in preference order.
/// Assembled from compiled features — adding an EP (ROCm, DirectML) is one
/// entry here plus a Cargo feature.
fn hardware_ep_cascade() -> Vec<(&'static str, ExecutionProviderDispatch)> {
    #[allow(unused_mut)]
    let mut cascade: Vec<(&'static str, ExecutionProviderDispatch)> = Vec::new();
    #[cfg(feature = "ort-cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;
        cascade.push((
            "CUDA",
            CUDAExecutionProvider::default().build().error_on_failure(),
        ));
    }
    #[cfg(feature = "ort-tensorrt")]
    {
        use ort::execution_providers::TensorRTExecutionProvider;
        cascade.push((
            "TensorRT",
            TensorRTExecutionProvider::default()
                .build()
                .error_on_failure(),
        ));
    }
    #[cfg(feature = "ort-coreml")]
    {
        use ort::execution_providers::CoreMLExecutionProvider;
        cascade.push((
            "CoreML",
            CoreMLExecutionProvider::default()
                .build()
                .error_on_failure(),
        ));
    }
    // Last of the hardware EPs: vendor-neutral (AMD/Intel/NVIDIA via
    // DX12/Vulkan/Metal) but experimental upstream, so a vendor-specific EP above
    // is always preferred when one is compiled in. Ranked below CoreML for the
    // same reason on macOS.
    #[cfg(feature = "ort-webgpu")]
    {
        use ort::execution_providers::WebGPUExecutionProvider;
        cascade.push((
            "WebGPU",
            WebGPUExecutionProvider::default()
                .build()
                .error_on_failure(),
        ));
    }
    cascade
}

pub struct OrtBackend {
    // ort's Session::run takes &mut self; EmbedBackend is &self. The accelerated
    // path is a single submit loop by design (GPU-aware parallelism), so a Mutex
    // costs nothing there; on the CPU path ORT parallelises internally via its
    // intra-op thread pool, so serialising submissions is also the right shape.
    session: Mutex<Session>,
    output_name: String,
    encoder: Encoder,
    /// "ort/CUDA", "ort/TensorRT", "ort/CoreML", or "ort/CPU".
    name: String,
    accelerated: bool,
}

impl OrtBackend {
    /// Try each compiled hardware EP in cascade order. Returns Ok(None) when no
    /// hardware EP confirms — never falls back to CPU here (the resolver owns
    /// that decision via the separate cpu() factory entry).
    fn load_accelerated(model_dir: &Path, desc: ModelDescriptor) -> Result<Option<Self>> {
        for (ep_name, dispatch) in hardware_ep_cascade() {
            match Self::load_with(model_dir, desc.clone(), Some((ep_name, dispatch))) {
                Ok(backend) => {
                    tracing::info!(ep = ep_name, "ORT hardware execution provider confirmed");
                    return Ok(Some(backend));
                }
                Err(e) => {
                    // Expected on machines without the hardware/runtime — log and
                    // try the next EP, then decline.
                    tracing::info!(ep = ep_name, "ORT EP not usable on this host: {e:#}");
                }
            }
        }
        Ok(None)
    }

    fn load_cpu(model_dir: &Path, desc: ModelDescriptor) -> Result<Self> {
        Self::load_with(model_dir, desc, None)
    }

    fn load_with(
        model_dir: &Path,
        desc: ModelDescriptor,
        ep: Option<(&'static str, ExecutionProviderDispatch)>,
    ) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let (ep_label, accelerated) = match &ep {
            Some((name, _)) => (*name, true),
            None => ("CPU", false),
        };
        tracing::info!(
            path = %model_path.display(),
            ep = ep_label,
            dim = desc.dim,
            n_inputs = desc.n_inputs,
            "loading ORT ONNX model"
        );

        let mut builder = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?;
        if let Some((_, dispatch)) = ep {
            // error_on_failure (set in the cascade) turns a failed EP
            // registration into a session-build error instead of ORT's silent
            // CPU fallback — this IS the "effective provider" confirmation.
            builder = builder
                .with_execution_providers([dispatch])
                .map_err(ort_err)?;
        }
        let session = builder
            .commit_from_file(&model_path)
            .map_err(ort_err)
            .context("build ORT session")?;

        anyhow::ensure!(
            session.inputs().len() == desc.n_inputs,
            "model.onnx expects {} inputs, model.toml n_inputs = {}",
            session.inputs().len(),
            desc.n_inputs
        );
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .context("ORT model has no outputs")?;

        let encoder = Encoder::load(model_dir, desc)?;
        let backend = Self {
            session: Mutex::new(session),
            output_name,
            encoder,
            name: format!("ort/{ep_label}"),
            accelerated,
        };

        // Warm-up inference: confirms the EP actually executes (some EPs fail
        // only at run time), pre-triggers kernel compilation, and validates the
        // descriptor's dim against the real output width.
        let wb = warmup_batch();
        let (flat, actual_seq) = backend.forward(&wb).context("ORT warm-up inference")?;
        anyhow::ensure!(
            flat.len() == wb.padded_batch * actual_seq * backend.encoder.desc().dim,
            "warm-up output width {} does not match model.toml dim {}",
            flat.len() / (wb.padded_batch * actual_seq),
            backend.encoder.desc().dim
        );
        tracing::info!(backend = %backend.name, "ORT session ready (warm-up passed)");
        Ok(backend)
    }

    fn bucket(&self) -> Option<&'static BucketSpec> {
        if self.accelerated {
            Some(&GPU_BUCKETS)
        } else {
            None
        }
    }

    /// Forward pass: `[padded_batch, seq]` i64 tensors → flat hidden states.
    fn forward(&self, tb: &TokenBatch) -> Result<(Vec<f32>, usize)> {
        let shape = [tb.padded_batch, tb.seq];
        let ids = Tensor::from_array((shape, tb.input_ids.clone())).map_err(ort_err)?;
        let mask = Tensor::from_array((shape, tb.attn_mask.clone())).map_err(ort_err)?;

        // Positional inputs, in the model's declared order — same 2-vs-3 input
        // contract as the tract path.
        let mut inputs: Vec<ort::session::SessionInputValue> = vec![ids.into(), mask.into()];
        if self.encoder.n_inputs() == 3 {
            let token_type = Tensor::from_array((shape, vec![0i64; tb.padded_batch * tb.seq]))
                .map_err(ort_err)?;
            inputs.push(token_type.into());
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("ORT session mutex poisoned"))?;
        let outputs = session.run(inputs.as_slice()).map_err(ort_err)?;
        let value = outputs
            .get(self.output_name.as_str())
            .with_context(|| format!("ORT output '{}' missing", self.output_name))?;
        let (out_shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(ort_err)
            .context("last_hidden_state as f32 tensor")?;
        anyhow::ensure!(
            out_shape.len() == 3,
            "expected [batch, seq, dim] output, got {} dims",
            out_shape.len()
        );
        let actual_seq = out_shape[1] as usize;
        Ok((data.to_vec(), actual_seq))
    }
}

/// Tiny fixed batch for the warm-up run: one row of [PAD] tokens, mask 1.
/// Token id 0 is a valid vocab entry for every supported tokenizer family.
fn warmup_batch() -> TokenBatch {
    let seq = 8;
    TokenBatch {
        batch: 1,
        padded_batch: 1,
        seq,
        input_ids: vec![0i64; seq],
        attn_mask: vec![1i64; seq],
    }
}

impl EmbedBackend for OrtBackend {
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<u8>>> {
        self.encoder
            .embed_documents_with(texts, self.bucket(), |tb| self.forward(tb))
    }

    fn embed_query(&self, text: &str) -> Result<Vec<u8>> {
        self.encoder
            .embed_query_with(text, self.bucket(), |tb| self.forward(tb))
    }

    fn backend_name(&self) -> &str {
        &self.name
    }

    fn is_accelerated(&self) -> bool {
        self.accelerated
    }

    // True even for ort/CPU, unlike the trait default. Every submission has to
    // take the session mutex (`Session::run` is `&mut self`), so N worker
    // threads would just queue behind it — N threads' overhead for one thread's
    // throughput. ORT's intra-op pool is what parallelises the CPU path.
    fn single_submitter(&self) -> bool {
        true
    }

    fn dim(&self) -> usize {
        self.encoder.dim
    }
}
