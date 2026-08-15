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
# Three outcomes, deliberately different:
#
#   ready      fixtures on disk. Prints ready=true; the caller runs the tests.
#   not set up fixtures release/assets absent, or checksums not recorded yet.
#              WARNS and prints ready=false so the caller SKIPS the tests. Not a
#              failure: an unconfigured fixture store is a missing capability,
#              not a defect in the code under test, and failing here would make
#              every PR red for a reason no reviewer can act on.
#   corrupt    an asset downloaded but its sha256 does not match. HARD FAILURE,
#              always. This is the one case that must never degrade to a skip —
#              running unverified model weights is exactly what the checksum is
#              there to prevent.
#
# Idempotent. Honours TRAVSR_EMBED_TEST_MODEL_DIR, which is also what the tests
# read, so point both at one directory.

set -uo pipefail

FIXTURE_TAG="${FIXTURE_TAG:-test-fixtures-v1}"
REPO="${FIXTURE_REPO:-Travsr-com/travsr-embed}"
DEST="${TRAVSR_EMBED_TEST_MODEL_DIR:-$HOME/.travsr/models}"

# name  archive  sha256-of-archive
# Replace the placeholders when the release assets are first uploaded. Until
# then this script reports ready=false and the model-backed tests are skipped.
FIXTURES=(
  "bge-small-en-v1.5 bge-small-en-v1.5.tar.gz SHA256_PLACEHOLDER_BGE"
  "tiny-modernbert   tiny-modernbert.tar.gz   SHA256_PLACEHOLDER_MODERNBERT"
)

READY=1

warn() {
  # ::warning:: renders in the GitHub UI; plain text elsewhere.
  if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::warning::$*"; else echo "WARNING: $*" >&2; fi
}

die() {
  if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::error::$*"; else echo "ERROR: $*" >&2; fi
  # ready=false as well as exit 1: belt and braces. If a caller ever ran this
  # with continue-on-error, the gate must still stop the tests from running
  # against a partially staged fixture directory.
  READY=0
  emit_ready
  exit 1
}

emit_ready() {
  local val="false"
  [ "$READY" -eq 1 ] && val="true"
  echo "fixtures ready: $val"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then echo "ready=$val" >> "$GITHUB_OUTPUT"; fi
}

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

  # Trust-on-first-use, deliberately. This is what makes the script idempotent —
  # re-running it does not re-download hundreds of MB, and CI restores the same
  # files from cache on every job. The cost is that an already-extracted fixture
  # is never re-verified: if a cache entry is corrupted after the fact, it stays
  # trusted. Accepted because the checksum guards the download path, which is
  # where an attacker-controlled substitution would enter, and the cache is
  # GitHub-scoped rather than public. Delete "$DEST/$name" to force re-fetch.
  if [ -f "$DEST/$name/model.onnx" ]; then
    echo "already present: $name (not re-verified — see note above)"
    continue
  fi

  case "$want" in
    SHA256_PLACEHOLDER_*)
      warn "Fixture '$name' has no recorded sha256 yet, so it cannot be verified and will not be downloaded. Model-backed tests will be SKIPPED. To enable them: upload $archive to the '$FIXTURE_TAG' release and put its sha256 in the FIXTURES table in .github/scripts/fetch-test-models.sh (see CONTRIBUTING.md)."
      READY=0
      continue
      ;;
  esac

  tmp="$(mktemp -d)"
  if ! gh release download "$FIXTURE_TAG" --repo "$REPO" --pattern "$archive" --dir "$tmp" 2>"$tmp/err"; then
    warn "Could not download '$archive' from the '$FIXTURE_TAG' release of $REPO: $(tr -d '\n' < "$tmp/err"). Model-backed tests will be SKIPPED."
    rm -rf "$tmp"
    READY=0
    continue
  fi

  got="$(sha256_of "$tmp/$archive")"
  if [ "$got" != "$want" ]; then
    rm -rf "$tmp"
    # Never a skip: a checksum mismatch means the bytes are not what we vetted.
    die "Checksum mismatch for $archive — expected $want, got $got. Refusing to use unverified model weights."
  fi

  mkdir -p "$DEST/$name"
  if ! tar -xzf "$tmp/$archive" -C "$DEST/$name" --strip-components=1; then
    rm -rf "$tmp"
    die "Failed to extract $archive."
  fi
  rm -rf "$tmp"

  if [ ! -f "$DEST/$name/model.onnx" ]; then
    die "$archive did not contain model.onnx."
  fi
  echo "staged: $name -> $DEST/$name"
done

emit_ready
