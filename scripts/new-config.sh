#!/usr/bin/env bash
# Scaffold a new benchmark config page in _configs/.
# The harness calls this (or writes the file directly) before a run, then fills
# in the result fields afterward.
#
# Usage:
#   scripts/new-config.sh \
#     --model openai/gpt-oss-20b --company OpenAI --family gpt-oss \
#     --params 20.9B --engine vLLM --quant MXFP4 --context 131072 \
#     --tags "gpt-oss,OpenAI,vLLM,MXFP4,MoE,20B"
set -euo pipefail
cd "$(dirname "$0")/.."

model="" company="" family="" params="" engine="" quant="" context="" tags="" modalities="text"
rationale="" source_repo="" download_url=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model="$2"; shift 2;;
    --company) company="$2"; shift 2;;
    --family) family="$2"; shift 2;;
    --params) params="$2"; shift 2;;
    --engine) engine="$2"; shift 2;;
    --quant) quant="$2"; shift 2;;
    --context) context="$2"; shift 2;;
    --tags) tags="$2"; shift 2;;
    --modalities) modalities="$2"; shift 2;;   # comma-separated: text,image,audio,video
    --rationale) rationale="$2"; shift 2;;     # why this quant was chosen
    --source-repo) source_repo="$2"; shift 2;; # HF repo the quant is downloaded from
    --download-url) download_url="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done
[ -n "$model" ] && [ -n "$engine" ] || { echo "need at least --model and --engine" >&2; exit 1; }

# Slug: model basename + engine + quant, lowercased, non-alnum -> dash.
slugify() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's#[^a-z0-9]+#-#g; s#^-+|-+$##g'; }
slug="$(slugify "${model##*/}-${engine}-${quant}")"
file="_configs/${slug}.md"

# Build YAML lists from comma-separated input.
tags_yaml="[$(echo "$tags" | sed 's/, */, /g')]"
modalities_yaml="[$(echo "$modalities" | sed 's/, */, /g')]"

cat > "$file" <<EOF
---
title: ${model##*/} · ${engine}${quant:+ · $quant}
model: ${model}
company: ${company}
family: ${family}
params: ${params}
engine: ${engine}
quant: ${quant}
quant_rationale: ${rationale}    # why THIS quant was chosen
source_repo: ${source_repo}      # HF repo the quant comes from (trusted)
download_url: ${download_url}    # link to its HF download page
context: ${context}
modalities: ${modalities_yaml}   # input modalities the MODEL accepts (detected from HF repo)
mm_served: true                  # set false if this run serves the model text-only
tags: ${tags_yaml}

status: pending                  # pending | blocked (needs human review) | done
prefill_toks:
decode_toks:
mem_gb:                          # peak; see mem_source
mem_source:                      # system MemAvailable delta (nvidia-smi/cgroup unusable on GB10)
completed_at:                    # date+time when status -> done
run_command: |
  # filled in by the run once it completes
---

EOF
echo "wrote $file"
