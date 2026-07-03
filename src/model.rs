// tract ONNX encoder backend — fully descriptor-driven (no hardcoded model params).
//
// Every model-specific value (embedding dim, pooling mode, query prefix, number of
// ONNX input tensors) is read from `<model_dir>/model.toml`, which the travsr CLI
// writes from the catalog at `travsr embed init`. This file contains NO per-model
// constants: the same code path runs BGE (standard BERT, 3 inputs, CLS, its own
// prefix) and arctic-embed-m (BERT, 2 inputs, CLS, a different prefix) purely by
// swapping the descriptor.
//
// Pipeline per batch:
//   tokenize (WordPiece/BPE, max 512, right-pad to batch-longest)
//   → tract ONNX run (2 or 3 inputs per descriptor.n_inputs)
//   → pooling (CLS = position 0, or masked MEAN) → [batch, dim]
//   → l2_normalize → pack as dim×f32 LE bytes

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tract_onnx::prelude::*;

const MAX_SEQ: usize = 512;

/// How the per-token hidden states are reduced to one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    /// Representation of the first token (position 0).
    Cls,
    /// Attention-mask-weighted average over tokens.
    Mean,
}

/// Per-model runtime descriptor, deserialized from `<model_dir>/model.toml`.
///
/// This is the single source of truth for model-specific behaviour — the sidecar
/// holds no hardcoded model table. The CLI writes it from the catalog entry.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelDescriptor {
    /// Output embedding dimension (e.g. 384, 768, 1024).
    pub dim: usize,
    /// Pooling mode.
    pub pooling: Pooling,
    /// Text prepended to queries only (documents get no prefix). May be empty.
    #[serde(default)]
    pub query_prefix: String,
    /// Number of ONNX input tensors: 2 = `input_ids, attention_mask`;
    /// 3 = classic BERT with `token_type_ids`.
    pub n_inputs: usize,
    /// Matryoshka truncation: if > 0, the native `dim`-length vector is sliced to its
    /// first `truncate_dim` values and re-normalized, and that shorter vector is what
    /// gets stored. 0 (default) = no truncation, store the full native vector.
    #[serde(default)]
    pub truncate_dim: usize,
}

impl ModelDescriptor {
    /// The stored embedding dimension: `truncate_dim` when truncating, else native `dim`.
    /// This is what the HNSW index, blob size, and reported `embedding_dim` all use.
    pub fn output_dim(&self) -> usize {
        if self.truncate_dim > 0 {
            self.truncate_dim
        } else {
            self.dim
        }
    }

    /// Read `<model_dir>/model.toml`. Errors (no hardcoded fallback) so a missing
    /// descriptor is a loud, actionable failure rather than silent wrong output.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("model.toml");
        let text = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "model descriptor not found: {}\n  Run `travsr embed init` to (re)write it.",
                path.display()
            )
        })?;
        let desc: ModelDescriptor =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        anyhow::ensure!(desc.dim > 0, "model.toml: dim must be > 0");
        anyhow::ensure!(
            desc.n_inputs == 2 || desc.n_inputs == 3,
            "model.toml: n_inputs must be 2 or 3 (got {})",
            desc.n_inputs
        );
        anyhow::ensure!(
            desc.truncate_dim <= desc.dim,
            "model.toml: truncate_dim ({}) must be <= dim ({})",
            desc.truncate_dim,
            desc.dim
        );
        Ok(desc)
    }
}

pub struct EncoderModel {
    // Arc so EncoderModel is cheaply Clone — multiple threads share one model load.
    model: Arc<TypedRunnableModel>,
    tokenizer: Tokenizer,
    desc: ModelDescriptor,
    /// Native hidden dim of the ONNX output (used to slice last_hidden_state rows).
    native_dim: usize,
    /// Stored/output dimension = `desc.output_dim()` (native, or truncated). This is
    /// what callers, the HNSW index, and blob sizes use.
    pub dim: usize,
}

impl Clone for EncoderModel {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            tokenizer: self.tokenizer.clone(),
            desc: self.desc.clone(),
            native_dim: self.native_dim,
            dim: self.dim,
        }
    }
}

impl EncoderModel {
    pub fn load(model_dir: &Path, desc: ModelDescriptor) -> Result<Self> {
        Self::load_inner(model_dir, desc)
    }

    pub fn load_for_shard(
        model_dir: &Path,
        _intra_threads: usize,
        desc: ModelDescriptor,
    ) -> Result<Self> {
        Self::load_inner(model_dir, desc)
    }

    fn load_inner(model_dir: &Path, desc: ModelDescriptor) -> Result<Self> {
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

        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".into(),
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("truncation config: {e}"))?;

        tracing::info!("tract model ready");
        let native_dim = desc.dim;
        let dim = desc.output_dim();
        // into_runnable() already returns Arc<TypedRunnableModel> — do not double-wrap.
        Ok(Self {
            model,
            tokenizer,
            desc,
            native_dim,
            dim,
        })
    }

    /// Embed texts for indexing (no prefix). Returns one `dim`×4-byte BLOB per input.
    pub fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<u8>>> {
        self.embed_raw(texts)
    }

    /// Embed a single query (with the descriptor's query prefix). Returns one BLOB.
    pub fn embed_query(&self, text: &str) -> Result<Vec<u8>> {
        let prefixed = format!("{}{}", self.desc.query_prefix, text);
        let mut blobs = self.embed_raw(&[prefixed.as_str()])?;
        blobs
            .pop()
            .ok_or_else(|| anyhow::anyhow!("empty embed result"))
    }

    fn embed_raw(&self, texts: &[&str]) -> Result<Vec<Vec<u8>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

        let batch = encodings.len();
        let seq = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1);

        let mut input_ids = vec![0i64; batch * seq];
        let mut attn_mask = vec![0i64; batch * seq];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            for j in 0..seq {
                input_ids[i * seq + j] = ids.get(j).copied().unwrap_or(0) as i64;
                attn_mask[i * seq + j] = mask.get(j).copied().unwrap_or(0) as i64;
            }
        }

        let t_ids = tract_ndarray::Array2::from_shape_vec((batch, seq), input_ids)?;
        let t_mask = tract_ndarray::Array2::from_shape_vec((batch, seq), attn_mask.clone())?;

        // Build the input tuple per descriptor.n_inputs. Classic BERT (bge) takes a
        // 3rd all-zero token_type_ids; 2-input exports (arctic-embed-m) must NOT be
        // handed a 3rd tensor or tract rejects the run.
        let mut inputs: TVec<TValue> =
            tvec![Tensor::from(t_ids).into(), Tensor::from(t_mask).into()];
        if self.desc.n_inputs == 3 {
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

        // Slice the ONNX output at the NATIVE hidden dim; truncate to the output dim
        // (Matryoshka) only after pooling, then re-normalize.
        let native = self.native_dim;
        let mut blobs = Vec::with_capacity(batch);
        for b in 0..batch {
            let pooled = match self.desc.pooling {
                Pooling::Cls => {
                    // CLS token at position 0.
                    let start = b * actual_seq * native;
                    flat[start..start + native].to_vec()
                }
                Pooling::Mean => {
                    // Attention-mask-weighted mean over tokens.
                    let mut acc = vec![0f32; native];
                    let mut count = 0f32;
                    for t in 0..actual_seq {
                        let m = attn_mask[b * seq + t];
                        if m == 0 {
                            continue;
                        }
                        let start = (b * actual_seq + t) * native;
                        let row = &flat[start..start + native];
                        for (a, &v) in acc.iter_mut().zip(row) {
                            *a += v;
                        }
                        count += 1.0;
                    }
                    if count > 0.0 {
                        for a in acc.iter_mut() {
                            *a /= count;
                        }
                    }
                    acc
                }
            };
            // Matryoshka: keep the first `self.dim` values (<= native), then l2-normalize.
            let normalized = l2_normalize(&pooled[..self.dim]);
            blobs.push(normalized.iter().flat_map(|&f| f.to_le_bytes()).collect());
        }

        Ok(blobs)
    }
}

/// Unpack a `dim`×4-byte BLOB into f32 values (little-endian).
pub fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}
