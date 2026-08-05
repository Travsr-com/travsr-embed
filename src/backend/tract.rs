// tract engine — pure-Rust CPU inference, the preferred engine for the
// architectures it can run (vectorized SIMD kernels, zero native deps,
// identical behaviour on macOS/Apple Silicon and OCI ARM64).
//
// tract executes a SUBSET of ONNX: standard BERT-style graphs. Architectures
// with RoPE / alternating local-global attention / GeGLU (ModernBERT,
// nomic-bert, …) fail at graph load — so `can_run` is a VALIDATED ALLOWLIST,
// and adding a family here means it has actually been run under tract, not
// that it merely might work.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use tract_onnx::prelude::*;

use super::{BackendFactory, EmbedBackend, TargetInfo, PREF_PREFERRED_CPU};
use crate::encode::{Encoder, TokenBatch};
use crate::model::ModelDescriptor;

/// Families proven to load + run under tract. ModernBERT et al. are simply
/// absent → the resolver filters tract out before it is ever tried.
const TRACT_FAMILIES: &[&str] = &["bert", "minilm"];

pub struct TractFactory;

impl BackendFactory for TractFactory {
    fn engine(&self) -> &'static str {
        "tract"
    }

    fn can_run(&self, family: &str) -> bool {
        TRACT_FAMILIES.contains(&family)
    }

    fn preference(&self, _target: &TargetInfo) -> i32 {
        PREF_PREFERRED_CPU
    }

    fn try_build(
        &self,
        model_dir: &Path,
        desc: &ModelDescriptor,
    ) -> Result<Option<Arc<dyn EmbedBackend>>> {
        Ok(Some(Arc::new(TractBackend::load(model_dir, desc.clone())?)))
    }
}

pub struct TractBackend {
    model: Arc<TypedRunnableModel>,
    encoder: Encoder,
}

impl TractBackend {
    pub fn load(model_dir: &Path, desc: ModelDescriptor) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        tracing::info!(
            path = %model_path.display(),
            dim = desc.dim,
            pooling = ?desc.pooling,
            n_inputs = desc.n_inputs,
            size_mb = std::fs::metadata(&model_path)
                .map(|m| m.len() / 1_048_576)
                .unwrap_or(0),
            "loading tract ONNX model"
        );

        let model = tract_onnx::onnx()
            .model_for_path(&model_path)
            .context("load model.onnx")?
            .into_optimized()
            .context("optimize ONNX graph")?
            .into_runnable()
            .context("make runnable plan")?;

        let encoder = Encoder::load(model_dir, desc)?;
        tracing::info!("tract model ready");
        // into_runnable() already returns Arc<TypedRunnableModel> — do not double-wrap.
        Ok(Self { model, encoder })
    }

    /// Forward pass: `[batch, seq]` i64 tensors → flat `last_hidden_state` f32.
    fn forward(&self, tb: &TokenBatch) -> Result<(Vec<f32>, usize)> {
        let batch = tb.padded_batch;
        let seq = tb.seq;
        let t_ids = tract_ndarray::Array2::from_shape_vec((batch, seq), tb.input_ids.clone())?;
        let t_mask = tract_ndarray::Array2::from_shape_vec((batch, seq), tb.attn_mask.clone())?;

        // Build the input tuple per descriptor.n_inputs. Classic BERT (bge) takes a
        // 3rd all-zero token_type_ids; 2-input exports (arctic-embed-m) must NOT be
        // handed a 3rd tensor or tract rejects the run.
        let mut inputs: TVec<TValue> =
            tvec![Tensor::from(t_ids).into(), Tensor::from(t_mask).into()];
        if self.encoder.n_inputs() == 3 {
            let token_type =
                tract_ndarray::Array2::from_shape_vec((batch, seq), vec![0i64; batch * seq])?;
            inputs.push(Tensor::from(token_type).into());
        }

        // Forward pass → output[0] = last_hidden_state [batch, seq, dim]
        let output = self.model.run(inputs)?;
        let actual_seq = output[0].shape()[1];
        let view = output[0].view();
        let flat: &[f32] = view
            .as_slice::<f32>()
            .context("last_hidden_state as f32 slice")?;
        Ok((flat.to_vec(), actual_seq))
    }
}

impl EmbedBackend for TractBackend {
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<u8>>> {
        self.encoder
            .embed_documents_with(texts, None, |tb| self.forward(tb))
    }

    fn embed_query(&self, text: &str) -> Result<Vec<u8>> {
        self.encoder
            .embed_query_with(text, None, |tb| self.forward(tb))
    }

    fn backend_name(&self) -> &str {
        "tract"
    }

    fn is_accelerated(&self) -> bool {
        false
    }

    fn dim(&self) -> usize {
        self.encoder.dim
    }
}
