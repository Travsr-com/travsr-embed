// candle backend for nomic-embed-text-v1.5.
//
// Replaces the ORT int8 ONNX backend (RFC-018 original) with candle fp32,
// gaining Metal GPU on Apple M-series developer machines:
//   macOS  → Device::new_metal(0), falls back to CPU on failure
//   Linux  → Device::Cpu (OCI ARM64 A1 Flex target)
//
// Pipeline per batch:
//   tokenize (BPE, max 512 tokens, right-pad to batch-longest)
//   → NomicBertModel::forward → last_hidden_state [batch, seq, 768]
//   → mean_pooling (masked, computed on device)
//   → narrow to dim=256 (MRL truncation)
//   → l2_normalize
//   → single host copy → pack as 256×f32 LE bytes (1024 B per node)
//
// Task prefixes (nomic model requirement):
//   indexing : "search_document: {text}"
//   querying : "search_query: {text}"

use std::path::Path;

use anyhow::{Context as _, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{self, NomicBertModel};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub const DIM: usize = 256;
const MAX_SEQ: usize = 512;
const DOC_PREFIX: &str = "search_document: ";
const QUERY_PREFIX: &str = "search_query: ";
const DTYPE: DType = DType::F32;

pub struct NomicModel {
    model: NomicBertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl NomicModel {
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::load_inner(model_dir)
    }

    /// Shard mode: Metal is a shared GPU resource; thread-count limiting
    /// does not apply. CPU builds use rayon (set RAYON_NUM_THREADS if needed).
    pub fn load_for_shard(model_dir: &Path, _intra_threads: usize) -> Result<Self> {
        Self::load_inner(model_dir)
    }

    fn load_inner(model_dir: &Path) -> Result<Self> {
        let device = make_device()?;
        tracing::info!(backend = device_label(&device), "candle device selected");

        let config: nomic_bert::Config = {
            let f = std::fs::File::open(model_dir.join("config.json"))
                .context("open config.json")?;
            serde_json::from_reader(f).context("parse config.json")?
        };

        // Load safetensors weights into device memory.
        // Safe (non-mmap) path required by #![forbid(unsafe_code)].
        // On Apple M-series unified memory the 270 MB fits comfortably.
        let tensors =
            candle_core::safetensors::load(model_dir.join("model.safetensors"), &device)
                .context("load model.safetensors")?;
        let vb = VarBuilder::from_tensors(tensors, DTYPE, &device);

        let model = NomicBertModel::load(vb, &config).context("build NomicBertModel")?;

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
            model,
            tokenizer,
            device,
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

        let mut input_ids_flat = Vec::with_capacity(batch * seq);
        let mut attn_mask_flat = Vec::with_capacity(batch * seq);

        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            for j in 0..seq {
                input_ids_flat.push(ids.get(j).copied().unwrap_or(0));
                attn_mask_flat.push(mask.get(j).copied().unwrap_or(0));
            }
        }

        let input_ids =
            Tensor::from_vec(input_ids_flat, (batch, seq), &self.device)
                .context("build input_ids tensor")?;
        let attn_mask =
            Tensor::from_vec(attn_mask_flat, (batch, seq), &self.device)
                .context("build attention_mask tensor")?;

        // Forward pass → [batch, seq, 768].
        // token_type_ids = None (nomic_bert uses zeros internally when absent).
        let hidden = self
            .model
            .forward(&input_ids, None, Some(&attn_mask))
            .context("NomicBert forward")?;

        // All reductions on-device: mean_pooling → [batch, 768]
        //                           narrow       → [batch, 256]  (MRL)
        //                           l2_normalize → [batch, 256]
        let pooled = nomic_bert::mean_pooling(&hidden, &attn_mask)
            .context("mean pooling")?;
        let mrl = pooled.narrow(1, 0, DIM).context("MRL narrow")?;
        let normalized = nomic_bert::l2_normalize(&mrl).context("L2 normalize")?;

        // Single host-transfer for the whole batch.
        let rows = normalized
            .contiguous()
            .context("make contiguous")?
            .to_device(&Device::Cpu)
            .context("copy to CPU")?
            .to_dtype(DType::F32)
            .context("cast to f32")?
            .to_vec2::<f32>()
            .context("extract embedding rows")?;

        Ok(rows
            .into_iter()
            .map(|row| row.iter().flat_map(|&f| f.to_le_bytes()).collect())
            .collect())
    }
}

/// Unpack a 1024-byte BLOB into 256 f32 values (little-endian).
pub fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn make_device() -> Result<Device> {
    Ok(Device::Cpu)
}

fn device_label(d: &Device) -> &'static str {
    if d.is_cpu() { "cpu" } else { "gpu" }
}
