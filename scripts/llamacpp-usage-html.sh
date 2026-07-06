#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-qwen-local-code}"
OUT_DIR="${OUT_DIR:-results/llamacpp-usage}"
INTERVAL="${INTERVAL:-30}"
WATCH="${WATCH:-0}"
RECENT_REQUESTS="${RECENT_REQUESTS:-30}"
RECENT_LOG_LINES="${RECENT_LOG_LINES:-160}"

mkdir -p "$OUT_DIR"

html_escape() {
  sed \
    -e 's/&/\&amp;/g' \
    -e 's/</\&lt;/g' \
    -e 's/>/\&gt;/g'
}

render_once() {
  local raw_log="$OUT_DIR/llamacpp.log"
  local rows="$OUT_DIR/requests.rows.html"
  local recent_log="$OUT_DIR/recent-log.html"
  local stats="$OUT_DIR/stats.env"
  local models_json="$OUT_DIR/models.json"
  local now status model_id

  now="$(date '+%Y-%m-%d %H:%M:%S %z')"
  status="$(docker ps --filter "name=^/${CONTAINER}$" --format '{{.Status}}' || true)"
  [[ -n "$status" ]] || status="not running"

  docker logs "$CONTAINER" >"$raw_log" 2>&1 || true
  curl -fsS http://127.0.0.1:8080/v1/models >"$models_json" 2>/dev/null || true
  model_id="$(sed -n 's/.*"id":"\([^"]*\)".*/\1/p' "$models_json" | head -1)"
  [[ -n "$model_id" ]] || model_id="unknown"

  awk -v rows="$rows" -v stats="$stats" -v recent="$RECENT_REQUESTS" '
    function ms_count(  i) {
      for (i = 1; i <= NF; i++) {
        if ($i == "ms") return $(i - 1) + 0
      }
      return 0
    }
    function token_count(  i) {
      for (i = 1; i <= NF; i++) {
        if ($i == "tokens" && $(i - 2) == "/") return $(i - 1) + 0
      }
      return 0
    }
    function tps_count(  i) {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^second\)?$/ && $(i - 1) == "per" && $(i - 2) == "tokens") return $(i - 3) + 0
      }
      return 0
    }
    function task_id(  i) {
      for (i = 1; i <= NF; i++) {
        if ($i == "task") return $(i + 1)
      }
      return "?"
    }
    /prompt eval time/ {
      task = task_id()
      prompt[task] = token_count()
      prompt_ms[task] = ms_count()
      prompt_tps[task] = tps_count()
      if (!(task in seen)) {
        seen[task] = 1
        order[++n] = task
      }
    }
    / eval time/ && !/prompt eval time/ {
      task = task_id()
      gen[task] = token_count()
      gen_ms[task] = ms_count()
      gen_tps[task] = tps_count()
    }
    /total time/ {
      total_ms[task_id()] = ms_count()
    }
    END {
      start = n - recent + 1
      if (start < 1) start = 1
      for (i = n; i >= start; i--) {
        task = order[i]
        p = prompt[task] + 0
        g = gen[task] + 0
        total = p + g
        total_seconds = (total_ms[task] + 0) / 1000
        printf "<tr><td>%s</td><td class=\"num\">%d</td><td class=\"num\">%d</td><td class=\"num\">%d</td><td class=\"num\">%.1f</td><td class=\"num\">%.1f</td><td class=\"num\">%.2f</td></tr>\n", task, p, g, total, prompt_tps[task] + 0, gen_tps[task] + 0, total_seconds > rows
      }
      close(rows)

      for (i = 1; i <= n; i++) {
        task = order[i]
        p = prompt[task] + 0
        g = gen[task] + 0
        total = p + g
        all_prompt += p
        all_gen += g
        all_prompt_ms += prompt_ms[task] + 0
        all_gen_ms += gen_ms[task] + 0
        if (p > max_prompt) max_prompt = p
        if (total > max_total) max_total = total
      }
      prefill_tps = all_prompt_ms > 0 ? all_prompt / (all_prompt_ms / 1000) : 0
      decode_tps = all_gen_ms > 0 ? all_gen / (all_gen_ms / 1000) : 0
      avg_total = n > 0 ? (all_prompt + all_gen) / n : 0
      printf "REQUESTS=%d\nPROMPT=%d\nGENERATED=%d\nTOTAL=%d\nPREFILL_TPS=%.1f\nDECODE_TPS=%.1f\nAVG_TOTAL=%.0f\nMAX_PROMPT=%d\nMAX_TOTAL=%d\n", n, all_prompt, all_gen, all_prompt + all_gen, prefill_tps, decode_tps, avg_total, max_prompt, max_total > stats
    }
  ' "$raw_log"

  # shellcheck disable=SC1090
  source "$stats"

  tail -n "$RECENT_LOG_LINES" "$raw_log" | html_escape >"$recent_log"

  cat >"$OUT_DIR/index.html" <<HTML
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="$INTERVAL">
  <title>llama.cpp Usage</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0c0f12;
      --panel: #15191e;
      --panel-2: #1b2027;
      --text: #e9eef4;
      --muted: #96a1ad;
      --line: #2a313a;
      --accent: #7dd3fc;
      --ok: #8bd17c;
      --warn: #f2c36b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }
    .sub {
      color: var(--muted);
      margin-top: 4px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--muted);
      white-space: nowrap;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--ok);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .metric strong {
      display: block;
      font-size: 24px;
      letter-spacing: 0;
    }
    section {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      margin-top: 14px;
      overflow: hidden;
    }
    section h2 {
      margin: 0;
      padding: 12px 14px;
      font-size: 15px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
      letter-spacing: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      padding: 9px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      background: #11161b;
    }
    tr:last-child td { border-bottom: 0; }
    .num {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    pre {
      margin: 0;
      padding: 12px 14px;
      overflow: auto;
      max-height: 520px;
      color: #d6dde5;
      background: #090b0e;
      font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    a { color: var(--accent); }
    @media (max-width: 760px) {
      main { width: min(100vw - 20px, 1180px); margin-top: 14px; }
      header { display: block; }
      .pill { margin-top: 10px; white-space: normal; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric strong { font-size: 20px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>llama.cpp Usage</h1>
        <div class="sub">$model_id via $CONTAINER · refreshed $now · every ${INTERVAL}s</div>
      </div>
      <div class="pill"><span class="dot"></span>$status</div>
    </header>

    <div class="grid">
      <div class="metric"><span>Completed Requests</span><strong>$REQUESTS</strong></div>
      <div class="metric"><span>Prompt Tokens</span><strong>$PROMPT</strong></div>
      <div class="metric"><span>Generated Tokens</span><strong>$GENERATED</strong></div>
      <div class="metric"><span>Total Tokens</span><strong>$TOTAL</strong></div>
      <div class="metric"><span>Prefill tok/s</span><strong>$PREFILL_TPS</strong></div>
      <div class="metric"><span>Decode tok/s</span><strong>$DECODE_TPS</strong></div>
      <div class="metric"><span>Avg Tokens / Request</span><strong>$AVG_TOTAL</strong></div>
      <div class="metric"><span>Max Prompt Tokens</span><strong>$MAX_PROMPT</strong></div>
    </div>

    <section>
      <h2>Recent Requests (Newest First)</h2>
      <table>
        <thead>
          <tr><th>Task</th><th class="num">Prompt</th><th class="num">Generated</th><th class="num">Total</th><th class="num">Prefill tok/s</th><th class="num">Decode tok/s</th><th class="num">Seconds</th></tr>
        </thead>
        <tbody>
$(cat "$rows")
        </tbody>
      </table>
    </section>

    <section>
      <h2>Recent Logs</h2>
      <pre>$(cat "$recent_log")</pre>
    </section>
  </main>
</body>
</html>
HTML
}

if [[ "$WATCH" == "1" || "$WATCH" == "true" ]]; then
  while true; do
    render_once
    sleep "$INTERVAL"
  done
else
  render_once
fi
