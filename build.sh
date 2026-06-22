#!/usr/bin/env bash
# One-off static build into ./_site using the same Dockerized Ruby toolchain.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="autobench-site"

# Lint config front matter first — a YAML break (e.g. an unquoted "key: value" colon-space inside a
# title/quant_rationale) makes Jekyll silently DROP that config's row from the listing. Fail loudly here.
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY' || { echo "!! Fix the broken front matter above before building."; exit 1; }
import glob, sys, yaml
bad=0
for p in sorted(glob.glob('_configs/*.md')):
    fm=open(p).read().split('---\n')
    if len(fm)<3:
        print(f"!! {p}: missing front matter"); bad+=1; continue
    try:
        yaml.safe_load(fm[1])
    except yaml.YAMLError as e:
        print(f"!! {p}: YAML error — {str(e).splitlines()[0]}"); bad+=1
sys.exit(1 if bad else 0)
PY
fi

docker build -f Dockerfile.site -t "$IMAGE" .
docker run --rm -v "$PWD":/site "$IMAGE" \
  bundle exec jekyll build
echo "==> Built into ./_site"
