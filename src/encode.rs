// Shared encode pipeline — tokenize / pool / truncate / blob-pack (issue #6).
//
// Every inference engine (tract, ort/CUDA, ort/CoreML, …) runs the SAME code
// for everything except the forward pass itself:
//
//   tokenize (WordPiece/BPE, max 512, right-pad)     ← here
//   → engine forward pass                            ← EmbedBackend impl
//   → pooling (CLS = position 0, or masked MEAN)     ← here
//   → Matryoshka truncate → l2_normalize             ← here
//   → pack as dim×f32 LE bytes                       ← here
//
// This is what makes cross-backend numerical parity hold: two backends can only
// differ by the matmul engine, never by tokenization or pooling arithmetic.
//
// Bucketed padding: accelerated backends (CUDA/TensorRT) re-optimize kernels
// per tensor shape, so `BucketSpec` rounds the sequence length up to a fixed
// bucket and the batch count up to a multiple. Padded rows carry attention
// mask 0 and are dropped before pooling, so bucketing never changes the output
// values — only the tensor shapes the engine sees. The CPU path passes no
// bucket spec and keeps today's exact BatchLongest shapes.

use std::path::Path;

use anyhow::Result;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::model::{ModelDescriptor, Pooling};

pub const MAX_SEQ: usize = 512;

/// Fixed-shape padding for engines that compile/optimize kernels per shape.
#[derive(Debug, Clone, Copy)]
pub struct BucketSpec {
    /// Sequence length is rounded up to the smallest bucket that fits.
    pub seq_buckets: &'static [usize],
    /// Batch count is rounded up to a multiple of this (padded rows get mask 0).
    pub batch_multiple: usize,
}

/// Default buckets for GPU execution providers. Five seq shapes × padded batch
/// counts keeps the number of distinct tensor shapes small enough that
/// CUDA/TensorRT re-optimisation happens a handful of times, not per batch.
#[cfg_attr(not(feature = "ort"), allow(dead_code))] // consumed by the ORT engine
pub const GPU_BUCKETS: BucketSpec = BucketSpec {
    seq_buckets: &[32, 64, 128, 256, MAX_SEQ],
    batch_multiple: 8,
};

impl BucketSpec {
    fn pad_seq(&self, seq: usize) -> usize {
        self.seq_buckets
            .iter()
            .copied()
            .find(|&b| b >= seq)
            .unwrap_or(MAX_SEQ)
    }

    fn pad_batch(&self, batch: usize) -> usize {
        batch.div_ceil(self.batch_multiple) * self.batch_multiple
    }
}

/// One tokenized batch, laid out as row-major `[padded_batch, seq]` i64 tensors.
/// Rows `batch..padded_batch` are all-zero padding (attention mask 0).
pub struct TokenBatch {
    /// Real input rows — only these produce output blobs.
    pub batch: usize,
    /// Rows in the tensors (== `batch` unless a `BucketSpec` padded it up).
    pub padded_batch: usize,
    /// Padded sequence width of the tensors.
    pub seq: usize,
    pub input_ids: Vec<i64>,
    pub attn_mask: Vec<i64>,
}

/// Everything shared between engines: tokenizer + descriptor-driven pooling,
/// truncation, and blob packing. Backends own an `Encoder` and provide only
/// the forward pass.
pub struct Encoder {
    tokenizer: Tokenizer,
    desc: ModelDescriptor,
    /// Native hidden dim of the ONNX output (used to slice last_hidden_state rows).
    native_dim: usize,
    /// Stored/output dimension = `desc.output_dim()` (native, or truncated). This is
    /// what callers, the HNSW index, and blob sizes use.
    pub dim: usize,
}

impl Encoder {
    pub fn load(model_dir: &Path, desc: ModelDescriptor) -> Result<Self> {
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

        let native_dim = desc.dim;
        let dim = desc.output_dim();
        Ok(Self {
            tokenizer,
            desc,
            native_dim,
            dim,
        })
    }

    #[cfg_attr(not(feature = "ort"), allow(dead_code))] // used by the ORT warm-up
    pub fn desc(&self) -> &ModelDescriptor {
        &self.desc
    }

    /// Tokenize a batch into `[padded_batch, seq]` i64 tensors. With no
    /// `bucket`, shapes are exactly BatchLongest (today's tract behaviour).
    pub fn tokenize(&self, texts: &[&str], bucket: Option<&BucketSpec>) -> Result<TokenBatch> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

        let batch = encodings.len();
        let longest = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1);

        let (seq, padded_batch) = match bucket {
            Some(b) => (b.pad_seq(longest), b.pad_batch(batch)),
            None => (longest, batch),
        };

        let mut input_ids = vec![0i64; padded_batch * seq];
        let mut attn_mask = vec![0i64; padded_batch * seq];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            for j in 0..seq {
                input_ids[i * seq + j] = ids.get(j).copied().unwrap_or(0) as i64;
                attn_mask[i * seq + j] = mask.get(j).copied().unwrap_or(0) as i64;
            }
        }

        Ok(TokenBatch {
            batch,
            padded_batch,
            seq,
            input_ids,
            attn_mask,
        })
    }

    /// Pool, truncate (Matryoshka), l2-normalize, and pack the engine's raw
    /// hidden states into one `dim`×4-byte LE BLOB per REAL input row.
    ///
    /// `flat` is `last_hidden_state` as `[padded_batch, actual_seq, native_dim]`
    /// row-major f32; `actual_seq` is the sequence length the engine reported.
    pub fn pool_pack(
        &self,
        flat: &[f32],
        actual_seq: usize,
        tb: &TokenBatch,
    ) -> Result<Vec<Vec<u8>>> {
        // The tokenized inputs and `attn_mask` are laid out at width `tb.seq`; the
        // pooling loops index hidden-state rows at width `actual_seq`. A standard
        // encoder always preserves the sequence length, so these are equal — assert
        // it so a model that ever reshapes the sequence fails loudly instead of
        // silently misaligning the mask.
        anyhow::ensure!(
            actual_seq == tb.seq,
            "model returned sequence length {actual_seq}, expected input length {}",
            tb.seq
        );
        // Slice the ONNX output at the NATIVE hidden dim; truncate to the output dim
        // (Matryoshka) only after pooling, then re-normalize.
        let native = self.native_dim;
        // Guard the descriptor's `dim` against the model's real output width: a wrong
        // `dim` in model.toml would otherwise silently mis-slice every row.
        anyhow::ensure!(
            flat.len() == tb.padded_batch * actual_seq * native,
            "hidden-state size {} != batch {} × seq {actual_seq} × dim {native}: \
             check model.toml `dim`",
            flat.len(),
            tb.padded_batch
        );
        let mut blobs = Vec::with_capacity(tb.batch);
        for b in 0..tb.batch {
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
                        let m = tb.attn_mask[b * actual_seq + t];
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

    /// Full document pipeline around a backend's forward pass. `forward` returns
    /// the raw hidden states and the sequence length the engine reported.
    pub fn embed_documents_with<F>(
        &self,
        texts: &[&str],
        bucket: Option<&BucketSpec>,
        forward: F,
    ) -> Result<Vec<Vec<u8>>>
    where
        F: FnOnce(&TokenBatch) -> Result<(Vec<f32>, usize)>,
    {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let tb = self.tokenize(texts, bucket)?;
        let (flat, actual_seq) = forward(&tb)?;
        self.pool_pack(&flat, actual_seq, &tb)
    }

    /// Full query pipeline: applies the descriptor's query prefix, then the
    /// document pipeline on the single prefixed text.
    pub fn embed_query_with<F>(
        &self,
        text: &str,
        bucket: Option<&BucketSpec>,
        forward: F,
    ) -> Result<Vec<u8>>
    where
        F: FnOnce(&TokenBatch) -> Result<(Vec<f32>, usize)>,
    {
        let prefixed = format!("{}{}", self.desc.query_prefix, text);
        let mut blobs = self.embed_documents_with(&[prefixed.as_str()], bucket, forward)?;
        blobs
            .pop()
            .ok_or_else(|| anyhow::anyhow!("empty embed result"))
    }

    /// The engine's expected input-token count for a warm-up inference.
    pub fn n_inputs(&self) -> usize {
        self.desc.n_inputs
    }
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_seq_rounds_up_to_fixed_shapes() {
        assert_eq!(GPU_BUCKETS.pad_seq(1), 32);
        assert_eq!(GPU_BUCKETS.pad_seq(32), 32);
        assert_eq!(GPU_BUCKETS.pad_seq(33), 64);
        assert_eq!(GPU_BUCKETS.pad_seq(200), 256);
        assert_eq!(GPU_BUCKETS.pad_seq(4096), MAX_SEQ);
    }

    #[test]
    fn bucket_batch_rounds_to_multiple() {
        assert_eq!(GPU_BUCKETS.pad_batch(1), 8);
        assert_eq!(GPU_BUCKETS.pad_batch(8), 8);
        assert_eq!(GPU_BUCKETS.pad_batch(9), 16);
    }

    #[test]
    fn l2_normalize_unit_norm() {
        let v = l2_normalize(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_passthrough() {
        assert_eq!(l2_normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
    }
}
