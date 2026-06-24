// Throughput benchmark: tract ONNX fp32 — single-thread vs multi-thread.
//
// Tests BAAI/bge-small-en-v1.5 via tract with realistic production batch
// sizes (64 texts / batch), measuring scaling from 1 → 8 threads.
// Each thread holds the same Arc<TypedRunnableModel>; tract's SimplePlan
// is Send + Sync so concurrent run() calls are safe and independent.
//
// Run: cargo run --release --bin bench-tract

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
};
use tract_onnx::prelude::*;

const MAX_SEQ: usize = 128;
const BATCH_CAP: usize = 64; // realistic production batch size
const N_TEXTS: usize = 2_000; // enough to split across 8 threads cleanly

fn synthetic_texts() -> Vec<String> {
    let templates = [
        "function: {}",
        "method: {}",
        "class: {}",
        "var: {}",
        "interface: {}",
        "type: {}",
        "go-pkg: {}",
    ];
    let names = [
        "parseManifest", "getNodeById", "GraphTraverser", "reconcileLoop",
        "validateCert", "PodController", "watchResources", "applyConfig",
        "deleteSecret", "listNamespaces", "createDeployment", "updateReplicas",
        "scaleDown", "healthCheck", "leaderElect", "syncCache",
        "evictPod", "admitRequest", "buildIndex", "resolveConflict",
        "mergeStrategy", "patchObject", "encodeToken", "decodeJWT",
        "rotateCerts", "flushBuffer", "drainQueue", "signalHandler",
        "shutdownGrace", "backoffRetry", "circuitBreak", "rateLimiter",
        "filterEvents", "aggregateMetrics", "reportStatus", "emitEvent",
        "watchdog", "reapZombies", "claimLease", "releaseLease",
        "compactLog", "snapshotState", "restoreState", "checkpointing",
        "propagateDeletion", "ownerReference", "garbageCollect", "markSweep",
        "injectSidecar", "extractLabels",
    ];
    (0..N_TEXTS)
        .map(|i| templates[i % templates.len()].replace("{}", names[i % names.len()]))
        .collect()
}

fn model_dir() -> PathBuf {
    dirs::home_dir()
        .expect("HOME not set")
        .join(".travsr/models/bge-small-en-v1.5")
}

fn load_tokenizer(dir: &PathBuf) -> Result<Tokenizer> {
    let mut tok = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    tok.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".into(),
    }));
    tok.with_truncation(Some(TruncationParams {
        max_length: MAX_SEQ,
        ..Default::default()
    }))
    .map_err(|e| anyhow::anyhow!("truncation: {e}"))?;
    Ok(tok)
}

fn load_tract_model(dir: &PathBuf) -> Result<Arc<TypedRunnableModel>> {
    let model_path = dir.join("model.onnx");
    println!(
        "loading tract model: {} ({:.0} MB)",
        model_path.display(),
        std::fs::metadata(&model_path)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0)
    );
    let t0 = Instant::now();
    let model = tract_onnx::onnx()
        .model_for_path(&model_path)
        .context("load onnx")?
        .into_optimized()
        .context("optimize")?
        .into_runnable()
        .context("make runnable")?;
    println!("  loaded + optimized in {:.2}s", t0.elapsed().as_secs_f32());
    Ok(model)
}

fn run_batch(
    model: &Arc<TypedRunnableModel>,
    tokenizer: &Tokenizer,
    texts: &[&str],
) -> Result<usize> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;

    let batch = encodings.len();
    let seq = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(1);

    let mut input_ids = vec![0i64; batch * seq];
    let mut attn_mask = vec![0i64; batch * seq];
    let token_type    = vec![0i64; batch * seq];

    for (i, enc) in encodings.iter().enumerate() {
        let ids  = enc.get_ids();
        let mask = enc.get_attention_mask();
        for j in 0..seq {
            input_ids[i * seq + j] = ids .get(j).copied().unwrap_or(0) as i64;
            attn_mask[i * seq + j] = mask.get(j).copied().unwrap_or(0) as i64;
        }
    }

    let t_ids  = tract_ndarray::Array2::from_shape_vec((batch, seq), input_ids )?;
    let t_mask = tract_ndarray::Array2::from_shape_vec((batch, seq), attn_mask )?;
    let t_type = tract_ndarray::Array2::from_shape_vec((batch, seq), token_type)?;

    model.run(tvec![
        Tensor::from(t_ids ).into(),
        Tensor::from(t_mask).into(),
        Tensor::from(t_type).into(),
    ])?;

    Ok(batch)
}

// Split texts into owned batches of at most BATCH_CAP texts each.
fn make_batches(texts: &[String]) -> Vec<Vec<String>> {
    texts
        .chunks(BATCH_CAP)
        .map(|c| c.to_vec())
        .collect()
}

fn bench_threads(
    model: &Arc<TypedRunnableModel>,
    tokenizer: &Tokenizer,
    batches: &Arc<Vec<Vec<String>>>,
    n_threads: usize,
) -> (f64, usize) {
    // Assign batches to threads round-robin.
    let mut thread_batches: Vec<Vec<usize>> = vec![Vec::new(); n_threads];
    for (i, _) in batches.iter().enumerate() {
        thread_batches[i % n_threads].push(i);
    }

    let t0 = Instant::now();
    std::thread::scope(|s| {
        let handles: Vec<_> = thread_batches
            .iter()
            .map(|indices| {
                let model_ref  = Arc::clone(model);
                let tok        = tokenizer.clone();
                let batches_ref = Arc::clone(batches);
                let idx_list: Vec<usize> = indices.clone();
                s.spawn(move || -> usize {
                    let mut done = 0usize;
                    for &bi in &idx_list {
                        let texts: Vec<&str> = batches_ref[bi].iter().map(String::as_str).collect();
                        done += run_batch(&model_ref, &tok, &texts).unwrap_or(0);
                    }
                    done
                })
            })
            .collect();

        let total: usize = handles.into_iter().map(|h| h.join().unwrap_or(0)).sum();
        (t0.elapsed().as_secs_f64(), total)
    })
}

fn main() -> Result<()> {
    let dir = model_dir();
    if !dir.join("model.onnx").exists() {
        anyhow::bail!(
            "model.onnx not found at {}\nDownload from BAAI/bge-small-en-v1.5 on HuggingFace",
            dir.display()
        );
    }

    let tokenizer = load_tokenizer(&dir)?;
    let model     = load_tract_model(&dir)?;

    let texts   = synthetic_texts();
    let batches = Arc::new(make_batches(&texts));
    println!(
        "\n{N_TEXTS} texts  |  {} batches of ≤{BATCH_CAP}  |  seq≤{MAX_SEQ}\n",
        batches.len()
    );

    // Warmup
    {
        let t: Vec<&str> = batches[0].iter().map(String::as_str).collect();
        run_batch(&model, &tokenizer, &t)?;
    }

    println!("{:<8}  {:>10}  {:>12}  {:>9}  {:>10}",
        "threads", "elapsed(s)", "nodes/sec", "per-node", "speedup");
    println!("{}", "─".repeat(56));

    let mut baseline_nps = 0f64;

    for &n in &[1usize, 2, 4, 8] {
        let (elapsed, total) = bench_threads(&model, &tokenizer, &batches, n);
        let nps     = total as f64 / elapsed;
        let per_ms  = elapsed * 1000.0 / total as f64;
        let speedup = if baseline_nps > 0.0 { nps / baseline_nps } else { 1.0 };
        if n == 1 { baseline_nps = nps; }
        println!("{:<8}  {:>10.3}  {:>12.0}  {:>7.2}ms  {:>9.2}×",
            n, elapsed, nps, per_ms, speedup);
    }

    println!("\n── candle fp32 baselines (observed, single-thread) ──");
    println!("  nomic fp32  (candle, NEON matmul)      : ~62  nodes/sec");
    println!("  BGE-small   (candle, scalar GeLU-ERF)  : ~21  nodes/sec");

    Ok(())
}
