#!/usr/bin/env bash
# Local Jekyll preview — Ruby lives entirely in Docker (nothing installed on the host).
# Builds the toolchain image once, then serves with live reload on http://localhost:4000
#
# Requires Docker access (be in the `docker` group, or run with sudo). See NOTES.md blocker #2.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="autobench-site"

echo "==> Building Jekyll toolchain image ($IMAGE)…"
docker build -f Dockerfile.site -t "$IMAGE" .

echo "==> Serving on http://localhost:4000  (Ctrl-C to stop)"
exec docker run --rm -it \
  -v "$PWD":/site \
  -p 4000:4000 -p 35729:35729 \
  "$IMAGE"
