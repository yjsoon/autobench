#!/usr/bin/env bash
# Run the standard sweep (sweep-gguf.sh: llama-bench + serving c1/c8/c32) for
# every model in a manifest TSV, one at a time, into one results/batch-<stamp>/ dir.
#
# Usage: batch-sweep.sh <manifest.tsv>
# Manifest columns: <gguf-relpath>	<label>	... (rest ignored here)
# Skips models whose GGUF is missing (logs it) so a partial download still runs.
set -u -o pipefail
cd "$(dirname "$0")/.."

MANIFEST="${1:?need a manifest tsv}"
MODELS_DIR="${MODELS_DIR:-$HOME/.lmstudio/models}"

STAMP="$(date '+%Y%m%d-%H%M%S')"
BATCH="results/batch-$STAMP"
mkdir -p "$BATCH"
ALL="$BATCH/summary-all.tsv"
LOG="$BATCH/batch.log"
printf 'model_label\tstarted_at\tfinished_at\tstatus\tlabel\tcommand\tresult\tmem\tvram_peak_gb\tvram_delta_gb\n' > "$ALL"

echo "batch start $(date '+%Y-%m-%d %H:%M:%S %z')" | tee "$LOG"

while IFS=$'\t' read -r relpath label _rest; do
  case "$relpath" in ''|\#*) continue;; esac
  label="${label:-$(basename "$relpath" .gguf)}"
  if [ ! -f "$MODELS_DIR/$relpath" ]; then
    echo "SKIP $label — missing $relpath" | tee -a "$LOG"
    continue
  fi
  echo "== [$(date '+%H:%M:%S')] START $label" | tee -a "$LOG"
  OUTDIR="$BATCH/$label" scripts/sweep-gguf.sh "$relpath" "$label" >>"$LOG" 2>&1 || \
    echo "!! $label sweep returned nonzero" | tee -a "$LOG"
  # Fold this model's rows into the combined summary.
  if [ -f "$BATCH/$label/summary.tsv" ]; then
    tail -n +2 "$BATCH/$label/summary.tsv" | sed "s/^/$label\t/" >> "$ALL"
  fi
  echo "== [$(date '+%H:%M:%S')] END $label" | tee -a "$LOG"
done < "$MANIFEST"

echo "batch done $(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "$LOG"
echo "combined summary: $ALL"