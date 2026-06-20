// ONNX inference + HuggingFace tokenizer for nomic-embed-text-v1.5 int8.
//
// Pipeline per batch:
//   tokenize (BPE, max 512 tokens, right-pad to batch-longest)
//   → ONNX int8 session → last_hidden_state [batch, seq, 768]
//   → mean-pool over real tokens (attention_mask == 1)
//   → MRL truncate to DIM=256
//   → L2 normalise
//   → pack as 256 × f32 little-endian bytes (1024 B per node)
//
// Task prefixes (nomic model requirement):
//   indexing : "search_document: {text}"
//   querying : "search_query: {text}"

use std::{path::Path, sync::Mutex};

use anyhow::{Context as _, Result};
use ort::{session::Session, value::Tensor};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub const DIM: usize = 256;
const HIDDEN: usize = 768;
const MAX_SEQ: usize = 512;
const DOC_PREFIX: &str = "search_document: ";
const QUERY_PREFIX: &str = "search_query: ";

pub struct NomicModel {
    // Session::run requires &mut self; Mutex provides interior mutability
    // while embed_batch / knn (from &self) can still call us.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl NomicModel {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let parallelism = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let session = Session::builder()
            .context("ORT session builder")?
            .with_intra_threads(parallelism)
            .map_err(|e| anyhow::anyhow!("setting intra-op thread count: {e}"))?
            .commit_from_file(model_dir.join("model_int8.onnx"))
            .context("loading model_int8.onnx")?;

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

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Embed texts for indexing.  Returns one 1024-byte BLOB per input.
    pub fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<u8>>> {
        let prefixed: Vec<String> = texts.iter().map(|t| format!("{DOC_PREFIX}{t}")).collect();
        self.embed_raw(&prefixed.iter().map(String::as_str).collect::<Vec<_>>())
    }

    /// Embed a single query text.  Returns 1024-byte BLOB.
    pub fn embed_query(&self, text: &str) -> Result<Vec<u8>> {
        let prefixed = format!("{QUERY_PREFIX}{text}");
        let mut blobs = self.embed_raw(&[&prefixed])?;
        blobs
            .pop()
            .ok_or_else(|| anyhow::anyhow!("empty embed result"))
    }

    // ── shared inference core ─────────────────────────────────────────────────

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

        // Pre-compute real token counts from encodings before building flat arrays,
        // so attn_mask_flat can be moved into the tensor without cloning.
        let n_real_per_item: Vec<usize> = encodings
            .iter()
            .map(|e| {
                e.get_attention_mask()
                    .iter()
                    .filter(|&&m| m == 1)
                    .count()
                    .max(1)
            })
            .collect();

        let mut input_ids_flat = vec![0i64; batch * seq];
        let mut attn_mask_flat = vec![0i64; batch * seq];
        let token_type_flat = vec![0i64; batch * seq]; // always 0 for single-sentence

        for (i, enc) in encodings.iter().enumerate() {
            for (j, (&id, &m)) in enc
                .get_ids()
                .iter()
                .zip(enc.get_attention_mask().iter())
                .enumerate()
            {
                input_ids_flat[i * seq + j] = id as i64;
                attn_mask_flat[i * seq + j] = m as i64;
            }
        }

        // Build ort 2.0 tensors from flat vecs; no ndarray feature required.
        // Shape ([batch, seq], Box<[T]>) uses the (D: ToShape, Box<[T]>) OwnedTensorArrayData impl.
        let ids_tensor = Tensor::from_array(([batch, seq], input_ids_flat.into_boxed_slice()))
            .context("build input_ids tensor")?;
        let mask_tensor = Tensor::from_array(([batch, seq], attn_mask_flat.into_boxed_slice()))
            .context("build attention_mask tensor")?;
        let types_tensor = Tensor::from_array(([batch, seq], token_type_flat.into_boxed_slice()))
            .context("build token_type_ids tensor")?;

        let mut guard = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("session mutex poisoned"))?;

        let outputs = guard
            .run(ort::inputs![
                "input_ids"      => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => types_tensor,
            ])
            .context("ONNX run")?;

        // try_extract_tensor returns (&Shape, &[f32]) — flat layout [batch * seq * HIDDEN].
        // Shape is [batch, seq, HIDDEN]; element at [i][j][k] = data[i*seq*HIDDEN + j*HIDDEN + k].
        let (_shape, hidden) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .context("extract last_hidden_state")?;

        let mut blobs = Vec::with_capacity(batch);
        for (i, &n_real) in n_real_per_item.iter().enumerate() {
            let mut pooled = [0f32; HIDDEN];
            for j in 0..n_real {
                let base = i * seq * HIDDEN + j * HIDDEN;
                for k in 0..HIDDEN {
                    pooled[k] += hidden[base + k];
                }
            }
            let scale = 1.0 / n_real as f32;
            pooled.iter_mut().for_each(|v| *v *= scale);

            // MRL truncation to DIM=256 then L2-normalise.
            let mut mrl = pooled[..DIM].to_vec();
            let norm: f32 = mrl.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
            mrl.iter_mut().for_each(|v| *v /= norm);

            // Pack as little-endian f32 bytes.
            let blob: Vec<u8> = mrl.iter().flat_map(|&f| f.to_le_bytes()).collect();
            blobs.push(blob);
        }

        Ok(blobs)
    }
}

/// Unpack a 1024-byte BLOB into 256 f32 values (little-endian).
pub fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
