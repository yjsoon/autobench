#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yingjie/Developer/amd-research/autobench"
MODEL_CONTAINER="qwen-local-code"
MODEL_URL="http://127.0.0.1:8080/v1/models"
DASHBOARD_URL="http://omarchy:8765/"
TAILSCALE_URL="http://100.107.214.67:8765/"

cd "$ROOT"

pause() {
  printf '\nPress Enter to continue... '
  read -r _
}

model_status() {
  printf '\nModel container:\n'
  docker ps --filter "name=^/${MODEL_CONTAINER}$" --format '  {{.Names}}  {{.Status}}  {{.Ports}}' || true

  printf '\nModel endpoint:\n'
  if curl -fsS "$MODEL_URL" >/tmp/llm-menu-models.json 2>/dev/null; then
    sed -n 's/.*"id":"\([^"]*\)".*/  model: \1/p' /tmp/llm-menu-models.json | head -1
    sed -n 's/.*"n_ctx":\([0-9]*\).*/  context: \1/p' /tmp/llm-menu-models.json | head -1
  else
    printf '  not reachable at %s\n' "$MODEL_URL"
  fi
}

dashboard_status() {
  printf '\nDashboard services:\n'
  systemctl --user --no-pager --plain status llamacpp-usage-html.service llamacpp-usage-http.service \
    | sed -n '/^●/p; /Active:/p' || true

  printf '\nDashboard URLs:\n'
  printf '  %s\n' "$DASHBOARD_URL"
  printf '  %s\n' "$TAILSCALE_URL"
}

token_summary() {
  scripts/llamacpp-usage-html.sh
  printf '\nToken summary:\n'
  sed -n '1,80p' results/llamacpp-usage/stats.env
  printf '\nDashboard: %s\n' "$DASHBOARD_URL"
}

start_model() {
  DETACH=1 scripts/serve-qwen-opencode.sh
}

stop_model() {
  docker stop "$MODEL_CONTAINER"
}

restart_model() {
  docker stop "$MODEL_CONTAINER" >/dev/null 2>&1 || true
  DETACH=1 scripts/serve-qwen-opencode.sh
}

start_dashboard() {
  systemctl --user start llamacpp-usage-html.service llamacpp-usage-http.service
}

stop_dashboard() {
  systemctl --user stop llamacpp-usage-http.service llamacpp-usage-html.service
}

restart_dashboard() {
  systemctl --user restart llamacpp-usage-html.service llamacpp-usage-http.service
}

open_dashboard() {
  printf 'Dashboard: %s\n' "$DASHBOARD_URL"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$DASHBOARD_URL" >/dev/null 2>&1 || true
  fi
}

while true; do
  clear 2>/dev/null || true
  cat <<MENU
Local LLM Control

Model:     ${MODEL_CONTAINER} on http://127.0.0.1:8080/v1
Dashboard: ${DASHBOARD_URL}

1) Status
2) Start model server
3) Stop model server
4) Restart model server
5) Tail model logs
6) Token summary / refresh dashboard
7) Start dashboard services
8) Stop dashboard services
9) Restart dashboard services
10) Open dashboard URL
q) Quit

MENU

  printf 'Choose: '
  read -r choice

  case "$choice" in
    1) model_status; dashboard_status; pause ;;
    2) start_model; pause ;;
    3) stop_model; pause ;;
    4) restart_model; pause ;;
    5) docker logs -f "$MODEL_CONTAINER" ;;
    6) token_summary; pause ;;
    7) start_dashboard; dashboard_status; pause ;;
    8) stop_dashboard; dashboard_status; pause ;;
    9) restart_dashboard; dashboard_status; pause ;;
    10) open_dashboard; pause ;;
    q|Q) exit 0 ;;
    *) printf 'Unknown choice: %s\n' "$choice"; pause ;;
  esac
done
