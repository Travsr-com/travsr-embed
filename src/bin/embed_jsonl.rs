// Offline text embedder for retrieval prototyping (#376 doc embeddings).
//
// Reads JSONL of {"id": "...", "text": "..."} and writes JSONL of
// {"id": "...", "dim": N, "b64": "<f32 LE base64>"}. Touches no database and no
// index: it is a pure text-in / vector-out tool so a retrieval prototype can be
// iterated outside the daemon.
//
// The `--mode` flag selects the model's asymmetric role. This matters: arctic and
// E5-family models are trained with a query-side prefix and a bare document side,
// and embedding a query without its prefix measurably degrades retrieval. Using the
// same code path as the sidecar (`EncoderModel::embed_documents` / `embed_query`)
// guarantees the prototype's vectors are byte-identical to production's.
//
// Run:
//   cargo run --release --bin embed-jsonl -- \
//     --model-dir ~/.travsr/models/arctic-embed-m-v1.5 \
//     --mode documents --in chunks.jsonl --out vectors.jsonl

use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};

// The shared encoder module, included by path rather than through a lib
// target (this crate is bin-only). This tool uses a subset of it, so the
// rest is dead code *here* while being live in the sidecar binary.
#[allow(dead_code)]
#[path = "../model.rs"]
mod model;

use model::{EncoderModel, ModelDescriptor};

/// Texts per ONNX forward pass. Matches the sidecar's realistic batch size; the
/// tokenizer right-pads to the batch-longest sequence, so mixing very short and
/// very long texts in one batch wastes compute. Inputs are sorted by length below.
const BATCH: usize = 32;

fn b64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Minimal string-field extractor. Avoids pulling serde_json into this bin for two
/// fields; the input is machine-generated, so escapes are limited to \" \\ \n \t \r.
fn json_str_field(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let q = rest.find('"')? + 1;
    let bytes = rest.as_bytes();
    let mut out = String::new();
    let mut i = q;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some(out),
            b'\\' => {
                i += 1;
                match bytes.get(i) {
                    Some(b'n') => out.push('\n'),
                    Some(b't') => out.push('\t'),
                    Some(b'r') => out.push('\r'),
                    Some(b'u') => {
                        let hex = rest.get(i + 1..i + 5)?;
                        let cp = u32::from_str_radix(hex, 16).ok()?;
                        out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        i += 4;
                    }
                    Some(&c) => out.push(c as char),
                    None => return None,
                }
            }
            _ => {
                // Copy the full UTF-8 sequence, not one byte.
                let ch = rest[i..].chars().next()?;
                out.push(ch);
                i += ch.len_utf8() - 1;
            }
        }
        i += 1;
    }
    None
}

fn main() -> Result<()> {
    let mut model_dir: Option<PathBuf> = None;
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut mode = String::from("documents");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                model_dir = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--in" => {
                input = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--out" => {
                output = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--mode" => {
                mode = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            other => bail!("unknown arg {other}"),
        }
    }

    let model_dir = model_dir.context("--model-dir is required")?;
    let input = input.context("--in is required")?;
    let output = output.context("--out is required")?;
    if mode != "documents" && mode != "queries" {
        bail!("--mode must be 'documents' or 'queries', got {mode:?}");
    }

    let desc = ModelDescriptor::load(&model_dir)
        .with_context(|| format!("loading model.toml from {}", model_dir.display()))?;
    let dim = desc.output_dim();
    eprintln!(
        "model {} dim={} mode={} query_prefix={:?}",
        model_dir.display(),
        dim,
        mode,
        desc.query_prefix
    );
    let enc = EncoderModel::load(&model_dir, desc).context("loading encoder")?;

    // Read all rows first so batches can be length-sorted (padding waste).
    let file = std::fs::File::open(&input).with_context(|| format!("open {}", input.display()))?;
    let mut rows: Vec<(String, String)> = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let id = json_str_field(&line, "id").context("row missing \"id\"")?;
        let text = json_str_field(&line, "text").context("row missing \"text\"")?;
        rows.push((id, text));
    }
    eprintln!("read {} rows", rows.len());

    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| rows[i].1.len());

    let out_file =
        std::fs::File::create(&output).with_context(|| format!("create {}", output.display()))?;
    let mut w = BufWriter::new(out_file);

    let started = std::time::Instant::now();
    let mut done = 0usize;
    // Collect results keyed by row index so output order matches input order.
    let mut vecs: Vec<Option<String>> = vec![None; rows.len()];

    for batch in order.chunks(BATCH) {
        let blobs = if mode == "queries" {
            // embed_query applies the descriptor prefix, one text at a time.
            let mut v = Vec::with_capacity(batch.len());
            for &idx in batch {
                v.push(enc.embed_query(&rows[idx].1).context("embed_query")?);
            }
            v
        } else {
            let texts: Vec<&str> = batch.iter().map(|&idx| rows[idx].1.as_str()).collect();
            enc.embed_documents(&texts).context("embed_documents")?
        };
        if blobs.len() != batch.len() {
            bail!(
                "backend returned {} blobs for {} texts",
                blobs.len(),
                batch.len()
            );
        }
        for (&idx, blob) in batch.iter().zip(blobs.iter()) {
            vecs[idx] = Some(b64(blob));
        }
        done += batch.len();
        if done.is_multiple_of(256) || done == rows.len() {
            let rate = done as f64 / started.elapsed().as_secs_f64();
            eprintln!("  {done}/{} ({rate:.0}/s)", rows.len());
        }
    }

    for (i, (id, _)) in rows.iter().enumerate() {
        let v = vecs[i].as_ref().context("missing vector for row")?;
        let esc = id.replace('\\', "\\\\").replace('"', "\\\"");
        writeln!(w, "{{\"id\":\"{esc}\",\"dim\":{dim},\"b64\":\"{v}\"}}")?;
    }
    w.flush()?;
    eprintln!(
        "wrote {} vectors to {} in {:.1}s",
        rows.len(),
        output.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
