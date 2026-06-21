#!/usr/bin/env bash
# One-off static build into ./_site using the same Dockerized Ruby toolchain.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="autobench-site"
docker build -f Dockerfile.site -t "$IMAGE" .
docker run --rm -v "$PWD":/site "$IMAGE" \
  bundle exec jekyll build
echo "==> Built into ./_site"
