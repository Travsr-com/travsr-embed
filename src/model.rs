// Model descriptor — fully descriptor-driven (no hardcoded model params).
//
// Every model-specific value (embedding dim, pooling mode, query prefix, number
// of ONNX input tensors, architecture family) is read from
// `<model_dir>/model.toml`, which the travsr CLI writes from the catalog at
// `travsr embed init`. This file contains NO per-model constants: the same code
// path runs BGE (standard BERT, 3 inputs, CLS, its own prefix) and
// arctic-embed-m (BERT, 2 inputs, CLS, a different prefix) purely by swapping
// the descriptor.
//
// The inference engine itself lives behind `backend::EmbedBackend` (issue #6):
// backend selection is a capability match on `family`, not a fixed engine.

use std::path::Path;

use anyhow::{Context as _, Result};

/// How the per-token hidden states are reduced to one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    /// Representation of the first token (position 0).
    Cls,
    /// Attention-mask-weighted average over tokens.
    Mean,
}

fn default_family() -> String {
    "bert".to_owned()
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
    /// Architecture family, e.g. "bert", "minilm", "modernbert", "nomic-bert".
    /// Written by the CLI from the catalog. Backend selection is a capability
    /// match on this tag: tract runs a validated allowlist, ORT runs everything.
    /// Defaults to "bert" so existing model.toml files need no migration.
    #[serde(default = "default_family")]
    pub family: String,
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
        Self::parse(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// Parse and validate a model.toml string.
    pub fn parse(text: &str) -> Result<Self> {
        let desc: ModelDescriptor = toml::from_str(text)?;
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
        anyhow::ensure!(
            !desc.family.is_empty(),
            "model.toml: family must be non-empty"
        );
        Ok(desc)
    }
}

/// Unpack a `dim`×4-byte BLOB into f32 values (little-endian).
pub fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    // as_chunks over chunks_exact: same semantics (trailing partial chunk
    // ignored), no per-chunk try_into, and what clippy 1.98's
    // chunks_exact_to_as_chunks lint asks for.
    blob.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_defaults_to_bert() {
        let desc = ModelDescriptor::parse(
            r#"
            dim = 384
            pooling = "cls"
            n_inputs = 3
            "#,
        )
        .unwrap();
        assert_eq!(desc.family, "bert");
    }

    #[test]
    fn family_tag_is_read() {
        let desc = ModelDescriptor::parse(
            r#"
            dim = 768
            pooling = "mean"
            n_inputs = 2
            family = "modernbert"
            "#,
        )
        .unwrap();
        assert_eq!(desc.family, "modernbert");
    }
}
