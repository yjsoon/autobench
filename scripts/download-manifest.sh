#!/usr/bin/env bash
# Download every GGUF listed in a manifest TSV to $MODELS_DIR, deriving the HF
# URL from the relpath (<org>/<repo>/<file> -> huggingface.co/<org>/<repo>).
#
# Usage: download-manifest.sh <manifest.tsv>
# Manifest: tab-separated, first column = gguf relpath under MODELS_DIR.
# Lines starting with # and blank lines are skipped. Resumable (curl -C -).
set -u -o pipefail

MANIFEST="${1:?need a manifest tsv}"
MODELS_DIR="${MODELS_DIR:-$HOME/.lmstudio/models}"

fail=0
while IFS=$'\t' read -r relpath _rest; do
  case "$relpath" in ''|\#*) continue;; esac
  dest="$MODELS_DIR/$relpath"
  repo="$(echo "$relpath" | cut -d/ -f1-2)"
  file="$(echo "$relpath" | cut -d/ -f3-)"
  url="https://huggingface.co/$repo/resolve/main/$file"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    echo "== have $relpath ($(du -h "$dest" | cut -f1))"
    continue
  fi
  echo "== fetch $relpath"
  if ! curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "$url"; then
    echo "!! FAILED $relpath" >&2
    fail=$((fail + 1))
  fi
done < "$MANIFEST"

echo "== download pass done (failures: $fail)"
exit "$fail"
