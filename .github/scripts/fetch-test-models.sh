#!/usr/bin/env bash
# Stage the model fixtures the #[ignore] backend tests need (issue #6).
#
# Two fixtures:
#   bge-small-en-v1.5  the model the sidecar actually ships against — used for
#                      tract-vs-ORT numerical parity and the CoreML check.
#   tiny-modernbert    a few-MB ModernBERT graph (RoPE + local/global attention
#                      + GeGLU). Proves the two halves of the capability design
#                      for real: tract CANNOT load it, and ORT runs it with no
#                      sidecar code beyond `family = "modernbert"`.
#
# Fixtures are pulled from a release in THIS repo, not from Hugging Face, so a
# CI run never depends on a third party's uptime or on a model being re-tagged
# upstream. See CONTRIBUTING.md for how to (re)build and upload them.
#
# Idempotent: verifies checksums and skips anything already present. Honours
# TRAVSR_EMBED_TEST_MODEL_DIR, which is also what the tests read.

set -euo pipefail

FIXTURE_TAG="${FIXTURE_TAG:-test-fixtures-v1}"
REPO="${FIXTURE_REPO:-Travsr-com/travsr-embed}"
DEST="${TRAVSR_EMBED_TEST_MODEL_DIR:-$HOME/.travsr/models}"

# name  archive  sha256-of-archive
# Fill the checksums in when the release assets are first uploaded; the script
# refuses to run with placeholders rather than silently skipping verification.
FIXTURES=(
  "bge-small-en-v1.5 bge-small-en-v1.5.tar.gz SHA256_PLACEHOLDER_BGE"
  "tiny-modernbert   tiny-modernbert.tar.gz   SHA256_PLACEHOLDER_MODERNBERT"
)

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

mkdir -p "$DEST"

for row in "${FIXTURES[@]}"; do
  # shellcheck disable=SC2086 # deliberate word-splitting of the table row
  set -- $row
  name="$1" archive="$2" want="$3"

  if [ -f "$DEST/$name/model.onnx" ]; then
    echo "✓ $name already present in $DEST"
    continue
  fi

  case "$want" in
    SHA256_PLACEHOLDER_*)
      echo "ERROR: no checksum recorded for '$name' yet." >&2
      echo "       Upload the fixture to the '$FIXTURE_TAG' release and put its" >&2
      echo "       sha256 in the FIXTURES table in this script. Refusing to" >&2
      echo "       download unverified model weights." >&2
      exit 1
      ;;
  esac

  echo "→ fetching $name"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT

  gh release download "$FIXTURE_TAG" \
    --repo "$REPO" \
    --pattern "$archive" \
    --dir "$tmp"

  got="$(sha256_of "$tmp/$archive")"
  if [ "$got" != "$want" ]; then
    echo "ERROR: checksum mismatch for $archive" >&2
    echo "  expected $want" >&2
    echo "  got      $got" >&2
    exit 1
  fi

  mkdir -p "$DEST/$name"
  tar -xzf "$tmp/$archive" -C "$DEST/$name" --strip-components=1
  test -f "$DEST/$name/model.onnx" || {
    echo "ERROR: $archive did not contain model.onnx" >&2
    exit 1
  }
  echo "✓ $name staged in $DEST/$name"

  rm -rf "$tmp"
  trap - EXIT
done

echo "fixtures ready in $DEST"
