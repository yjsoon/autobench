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

# Bind explicitly on all host interfaces so the preview is reachable from the LAN.
# Safe: the site is a read-only static preview.
lan_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "==> Serving on all interfaces (Ctrl-C to stop):"
echo "      http://localhost:4000/autobench/"
[ -n "$lan_ip" ] && echo "      http://$lan_ip:4000/autobench/   (LAN)"
exec docker run --rm -it \
  -v "$PWD":/site \
  -p 0.0.0.0:4000:4000 -p 0.0.0.0:35729:35729 \
  "$IMAGE"
