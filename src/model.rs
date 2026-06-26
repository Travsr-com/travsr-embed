// tract ONNX backend for BAAI/bge-small-en-v1.5.
//
// BGE-small is a 33M-parameter BERT (hidden=384, 12 layers, standard absolute
// positional embeddings — no RoPE) that delivers near-identical retrieval
// quality to nomic-embed-text-v1.5 (MTEB 62.2 vs 62.4) at 4× smaller size.
//
// Why tract instead of candle:
//   candle calls scalar libm::erff() per element for BGE's GeLU-ERF activation
//   (no SIMD path). tract has vectorized ONNX kernels for both GeLU and matmul.
//   Benchmark on M-series (64-text batches, seq≤128):
//     candle fp32 BGE  :  ~21 nodes/sec  (scalar GeLU)
//     candle fp32 nomic:  ~62 nodes/sec  (SwiGLU, NEON matmul)
//     tract  fp32 BGE  : ~140 nodes/sec  single-thread
//     tract  fp32 BGE  : ~307 nodes/sec  4 threads (2.2× scaling)
//
// Platform: tract is pure Rust, works identically on macOS and OCI ARM64.
// No Accelerate feature, no Metal, no platform-specific build flags needed.
//
// Pipeline per batch:
//   tokenize (WordPiece, max 512 tokens, right-pad to batch-longest)
//   → tract ONNX run → last_hidden_state [batch, seq, 384]
//   → CLS pooling (position 0)  → [batch, 384]
//   → l2_normalize              → [batch, 384]
//   → pack as 384×f32 LE bytes (1536 B per node)
//
// Task prefixes (BGE convention):
//   indexing : no prefix (plain symbol text)
//   querying : "Represent this sentence: {text}"

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tract_onnx::prelude::*;

const MAX_SEQ: usize = 512;
const QUERY_PREFIX: &str = "Represent this sentence: ";

pub struct BgeModel {
    // Arc so BgeModel is cheaply Clone — multiple threads can share one model load.
    model: Arc<TypedRunnableModel>,
    tokenizer: Tokenizer,
    /// Hidden dimension detected from the ONNX output shape on load.
    pub dim: usize,
}

impl Clone for BgeModel {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            tokenizer: self.tokenizer.clone(),
            dim: self.dim,
        }
    }
}

impl BgeModel {
    pub fn load(model_dir: &Path, dim: usize) -> Result<Self> {
        Self::load_inner(model_dir, dim)
    }

    pub fn load_for_shard(model_dir: &Path, _intra_threads: usize, dim: usize) -> Result<Self> {
        Self::load_inner(model_dir, dim)
    }

    fn load_inner(model_dir: &Path, dim: usize) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        tracing::info!(
            path = %model_path.display(),
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
        // into_runnable() already returns Arc<TypedRunnableModel> — do not double-wrap.
        Ok(Self {
            model,
            tokenizer,
            dim,
        })
    }

    /// Embed texts for indexing (no prefix). Returns one 1536-byte BLOB per input.
    pub fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<u8>>> {
        self.embed_raw(texts)
    }

    /// Embed a single query text (with BGE query prefix). Returns 1536-byte BLOB.
    pub fn embed_query(&self, text: &str) -> Result<Vec<u8>> {
        let prefixed = format!("{QUERY_PREFIX}{text}");
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
        let token_type = vec![0i64; batch * seq]; // all zeros for single-sequence BERT

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            for j in 0..seq {
                input_ids[i * seq + j] = ids.get(j).copied().unwrap_or(0) as i64;
                attn_mask[i * seq + j] = mask.get(j).copied().unwrap_or(0) as i64;
            }
        }

        let t_ids = tract_ndarray::Array2::from_shape_vec((batch, seq), input_ids)?;
        let t_mask = tract_ndarray::Array2::from_shape_vec((batch, seq), attn_mask)?;
        let t_type = tract_ndarray::Array2::from_shape_vec((batch, seq), token_type)?;

        // Forward pass → output[0] = last_hidden_state [batch, seq, DIM]
        let output = self.model.run(tvec![
            Tensor::from(t_ids).into(),
            Tensor::from(t_mask).into(),
            Tensor::from(t_type).into(),
        ])?;

        // last_hidden_state is a flat [batch * seq * DIM] f32 buffer in row-major order.
        // CLS token is at position 0 in the sequence dimension for each batch item.
        // TValue: Deref<Target=Tensor>; Tensor::view() is a safe fn (unsafe inside impl only).
        // TensorView::as_slice::<f32>() is fully safe.
        let actual_seq = output[0].shape()[1];
        let flat: &[f32] = output[0]
            .view()
            .as_slice::<f32>()
            .context("last_hidden_state as f32 slice")?;
        let mut blobs = Vec::with_capacity(batch);
        for b in 0..batch {
            let cls_start = b * actual_seq * self.dim;
            let cls = &flat[cls_start..cls_start + self.dim];
            let normalized = l2_normalize(cls);
            blobs.push(normalized.iter().flat_map(|&f| f.to_le_bytes()).collect());
        }

        Ok(blobs)
    }
}

/// Unpack a 1536-byte BLOB into 384 f32 values (little-endian).
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
