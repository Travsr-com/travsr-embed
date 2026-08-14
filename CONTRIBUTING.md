# Contributing

## Merge order with `travsr`

This repo path-depends on `travsr-plugin-protocol` and `travsr-plugin-sdk`
from `Travsr-com/travsr@master` (CI checks out the sibling repo and symlinks
it in - see `.github/workflows/ci.yml`).

If a PR here needs a travsr-side protocol change, that change must be merged
to `travsr` master **first**. Until then, CI on this PR stays red with an
unresolved-import error (the preflight step names the exact travsr SHA it
compiled against, so this is a one-look diagnosis, not a debugging session).
That is expected, not a bug in this repo - land the paired travsr PR, then
re-run CI here.

## Building the ORT (GPU) features locally

The default build is tract-only and needs nothing special. The `ort*` features
pull in ONNX Runtime, which `ort` downloads as a prebuilt binary at build time -
and that download is per-target, so where you can build what is not uniform:

| Host | What works |
| --- | --- |
| macOS (arm64/x86_64) | `--features ort`, `--features ort-coreml`. CoreML is in every macOS prebuilt, statically linked, no extra install. |
| Linux x86_64 | `--features ort`, `--features ort-cuda` (running it needs a host CUDA runtime + cuDNN; building it does not). |
| Linux aarch64 | `--features ort` only - no GPU prebuilt exists for this target. |
| Windows **MSVC** | `--features ort` builds. `--features ort-cuda` does **not**: ORT's prebuilt wants the dynamic CRT, this repo forces `/MT` for `esaxx-rs`, and the link fails with `LNK2019` on `__imp_*` symbols. Needs the MSVC C++ build tools installed either way. |
| Windows **GNU** | **Nothing.** `ort` publishes no prebuilt for `x86_64-pc-windows-gnu` and the build fails in `ort-sys` with "does not provide prebuilt binaries". |

So on Windows, check which toolchain is actually active (`rustup show`) before
concluding an ORT build is broken - a `-gnu` default host is the usual answer.
Pass `--target x86_64-pc-windows-msvc` (and have the MSVC toolset installed), or
do ORT work on Linux/macOS and let CI cover Windows.

Windows also needs the two defines CI sets, or `usearch` fails to compile:

```bash
export CFLAGS="-DSIMSIMD_TARGET_SAPPHIRE=0 -DMAP_FAILED=((void*)-1)"
export CXXFLAGS="$CFLAGS"
```

### Type-checking `src/backend/ort.rs` without a prebuilt

`ort`'s `load-dynamic` feature switches it to loading ONNX Runtime at run time,
which skips the build-time download entirely. That is enough to compile and lint
the ORT backend on a host with no prebuilt available:

```bash
cargo clippy --all-targets --features ort,ort/load-dynamic
```

Do not use it to *run* anything - there is no library to load. CI covers the real
builds (see the `ort` matrix rows and the `ort-cuda-link` job in `ci.yml`).

## Model fixtures for the `#[ignore]` tests

Several backend tests need real model weights, so they are `#[ignore]` by default:
cross-backend numerical parity, the ModernBERT capability proof, and the CoreML
confirmation check. Stage the fixtures once:

```bash
bash .github/scripts/fetch-test-models.sh
cargo test --features ort -- --ignored
```

The script reads `TRAVSR_EMBED_TEST_MODEL_DIR` (default `~/.travsr/models`) - the
same variable the tests use, so point both at one directory. It pulls from a
`test-fixtures-v1` release in *this* repo rather than Hugging Face, so CI never
depends on a third party's uptime or on an upstream re-tag.

Two fixtures, and what each is for:

- **`bge-small-en-v1.5`** - the model the sidecar actually ships against. Used for
  tract-vs-ORT parity (cosine >= 0.999) and the CoreML check.
- **`tiny-modernbert`** - a few-MB ModernBERT graph (RoPE + alternating
  local/global attention + GeGLU). It proves *both* halves of the capability
  design on a real graph: `tract` cannot load it at all, and ORT runs it with no
  sidecar code beyond `family = "modernbert"` in its `model.toml`. A
  BERT graph re-tagged as `modernbert` cannot prove either half, which is why a
  real one is required.

To rebuild the ModernBERT fixture: export `hf-internal-testing/tiny-random-ModernBertModel`
to ONNX (via `optimum`), pair it with its `tokenizer.json` and a hand-written
`model.toml` (`family = "modernbert"`, matching `dim`, mean pooling), verify it
loads under ORT locally, then upload it as a `.tar.gz` to the fixtures release and
put its sha256 in the `FIXTURES` table in the fetch script. The script refuses to
download anything whose checksum is not recorded.

## Verifying a GPU release asset by hand

CI proves the GPU builds compile, link, and report the right capabilities. It
cannot prove they *use the GPU* - no CI runner has one. Do this once per release on
real hardware, per accelerator.

**NVIDIA / `-cuda` asset:**

1. Download the asset plus its `libonnxruntime_providers_*.so` files; verify every
   `.sha256`.
2. `./travsr-embed-… --capabilities` - expect `"accelerated_compiled":true`.
3. `RUST_LOG=info` a reindex. Expect log lines for `ep=CUDA` **confirmed**,
   `backend=ort/CUDA`, and the single-submit-loop notice.
4. `nvidia-smi` shows utilization during the run, and `embed.db`'s `meta` table
   records `embed_backend = ort/CUDA`.
5. Vectors agree with the default (tract) asset: cosine >= 0.99 over ~100 nodes.
6. Cancel mid-reindex via the sentinel - workers must drain, not be killed.
7. **Move the provider `.so` files away and rerun.** The CUDA EP must decline, the
   resolver must fall back to tract, and the log must say so. This is what proves
   the `$ORIGIN` rpath bundling is doing anything - step 3 passes either way if the
   host happens to have a system-wide ONNX Runtime.

**AMD / Intel (WebGPU, once shipped):** steps 1-5 with `backend=ort/WebGPU` and
vendor GPU utilization (Task Manager, `radeontop`), plus a run on a machine with no
usable GPU to confirm it declines cleanly to tract instead of erroring.
