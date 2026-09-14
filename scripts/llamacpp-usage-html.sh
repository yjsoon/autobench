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
  local daily_rows="$OUT_DIR/daily.rows.html"
  local hourly_rows="$OUT_DIR/hourly.rows.html"
  local day_filters="$OUT_DIR/day-filters.html"
  local recent_log="$OUT_DIR/recent-log.html"
  local stats="$OUT_DIR/stats.env"
  local models_json="$OUT_DIR/models.json"
  local now status model_id

  now="$(date '+%Y-%m-%d %H:%M:%S %z')"
  status="$(docker ps --filter "name=^/${CONTAINER}$" --format '{{.Status}}' || true)"
  [[ -n "$status" ]] || status="not running"

  docker logs --timestamps "$CONTAINER" >"$raw_log" 2>&1 || true
  curl -fsS http://127.0.0.1:8080/v1/models >"$models_json" 2>/dev/null || true
  model_id="$(sed -n 's/.*"id":"\([^"]*\)".*/\1/p' "$models_json" | head -1)"
  [[ -n "$model_id" ]] || model_id="unknown"

  python3 - "$raw_log" "$rows" "$daily_rows" "$hourly_rows" "$day_filters" "$stats" "$RECENT_REQUESTS" <<'PY'
import html
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

raw_log, rows_path, daily_path, hourly_path, filters_path, stats_path, recent_s = sys.argv[1:]
recent = int(recent_s)

ts_re = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+(.*)$')
task_re = re.compile(r'\|\s+task\s+(-?\d+)\s+\|')
tokens_re = re.compile(r'/\s*([0-9]+)\s+tokens')
ms_re = re.compile(r'=\s*([0-9.]+)\s+ms')
tps_re = re.compile(r',\s*([0-9.]+)\s+tokens per second\)')

def parse_ts(line):
    match = ts_re.match(line)
    if not match:
        return None, line
    stamp = match.group(1)
    body = match.group(2)
    if '.' in stamp:
        base, frac = stamp[:-1].split('.', 1)
        frac = (frac + '000000')[:6]
        stamp = f'{base}.{frac}Z'
        dt = datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%S.%fZ')
    else:
        dt = datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ')
    return dt.replace(tzinfo=timezone.utc).astimezone(), body

def task_id(line):
    match = task_re.search(line)
    return match.group(1) if match else None

def token_count(line):
    match = tokens_re.search(line)
    return int(match.group(1)) if match else 0

def ms_count(line):
    match = ms_re.search(line)
    return float(match.group(1)) if match else 0.0

def tps_count(line):
    match = tps_re.search(line)
    return float(match.group(1)) if match else 0.0

def fmt_int(value):
    return f'{int(round(value)):,}'

def fmt_float(value, digits=1):
    return f'{value:,.{digits}f}'

def bar(value, max_value):
    pct = 0 if max_value <= 0 else max(2, min(100, (value / max_value) * 100))
    return f'<span class="usage-bar"><span style="width:{pct:.1f}%"></span></span>'

tasks = {}
order = []
with open(raw_log, errors='replace') as f:
    for original in f:
        dt, line = parse_ts(original.rstrip('\n'))
        task = task_id(line)
        if not task:
            continue
        rec = tasks.setdefault(task, {'task': task})
        if task not in order and 'prompt' not in rec:
            order.append(task)
        if 'prompt eval time' in line:
            rec['time'] = dt
            rec['prompt'] = token_count(line)
            rec['prompt_ms'] = ms_count(line)
            rec['prompt_tps'] = tps_count(line)
        elif ' eval time' in line and 'prompt eval time' not in line:
            rec['gen'] = token_count(line)
            rec['gen_ms'] = ms_count(line)
            rec['gen_tps'] = tps_count(line)
        elif 'total time' in line:
            rec['total_ms'] = ms_count(line)

complete = [
    rec for rec in tasks.values()
    if rec.get('prompt', 0) > 0 and ('gen' in rec or 'total_ms' in rec)
]
complete.sort(key=lambda r: (r.get('time') or datetime.min.astimezone(), int(r['task'])))

with open(rows_path, 'w') as out:
    for rec in reversed(complete[-recent:]):
        p = rec.get('prompt', 0)
        g = rec.get('gen', 0)
        total = p + g
        seconds = rec.get('total_ms', 0.0) / 1000
        when = rec['time'].strftime('%Y-%m-%d %H:%M:%S') if rec.get('time') else 'unknown'
        out.write(
            '<tr>'
            f'<td class="nowrap">{html.escape(when)}</td>'
            f'<td>{html.escape(rec["task"])}</td>'
            f'<td class="num">{fmt_int(p)}</td>'
            f'<td class="num">{fmt_int(g)}</td>'
            f'<td class="num">{fmt_int(total)}</td>'
            f'<td class="num">{fmt_float(rec.get("prompt_tps", 0.0))}</td>'
            f'<td class="num">{fmt_float(rec.get("gen_tps", 0.0))}</td>'
            f'<td class="num">{fmt_float(seconds, 2)}</td>'
            '</tr>\n'
        )

def blank_bucket():
    return {
        'requests': 0,
        'prompt': 0,
        'gen': 0,
        'prompt_ms': 0.0,
        'gen_ms': 0.0,
        'total_ms': 0.0,
        'max_prompt': 0,
        'max_total': 0,
    }

daily = defaultdict(blank_bucket)
hourly = defaultdict(blank_bucket)
for rec in complete:
    dt = rec.get('time')
    if not dt:
        continue
    day = dt.strftime('%Y-%m-%d')
    hour = dt.strftime('%Y-%m-%d %H:00')
    for bucket in (daily[day], hourly[hour]):
        p = rec.get('prompt', 0)
        g = rec.get('gen', 0)
        total = p + g
        bucket['requests'] += 1
        bucket['prompt'] += p
        bucket['gen'] += g
        bucket['prompt_ms'] += rec.get('prompt_ms', 0.0)
        bucket['gen_ms'] += rec.get('gen_ms', 0.0)
        bucket['total_ms'] += rec.get('total_ms', 0.0)
        bucket['max_prompt'] = max(bucket['max_prompt'], p)
        bucket['max_total'] = max(bucket['max_total'], total)

max_daily_total = max((b['prompt'] + b['gen'] for b in daily.values()), default=0)
max_hourly_total = max((b['prompt'] + b['gen'] for b in hourly.values()), default=0)

with open(filters_path, 'w') as out:
    out.write('<button class="filter active" type="button" data-day="all">All</button>\n')
    for day in sorted(daily.keys(), reverse=True):
        out.write(f'<button class="filter" type="button" data-day="{html.escape(day)}">{html.escape(day)}</button>\n')

def rates(bucket):
    prefill = bucket['prompt'] / (bucket['prompt_ms'] / 1000) if bucket['prompt_ms'] else 0.0
    decode = bucket['gen'] / (bucket['gen_ms'] / 1000) if bucket['gen_ms'] else 0.0
    return prefill, decode

with open(daily_path, 'w') as out:
    for day in sorted(daily.keys(), reverse=True):
        bucket = daily[day]
        total = bucket['prompt'] + bucket['gen']
        prefill, decode = rates(bucket)
        avg = total / bucket['requests'] if bucket['requests'] else 0.0
        day_hours = [(hour, hourly[hour]) for hour in hourly if hour.startswith(day)]
        peak_hour, peak_bucket = max(day_hours, key=lambda item: item[1]['prompt'] + item[1]['gen'])
        peak_total = peak_bucket['prompt'] + peak_bucket['gen']
        out.write(
            f'<button class="day-card" type="button" data-day-jump="{html.escape(day)}">'
            '<span class="day-card-top">'
            f'<span class="day-label">{html.escape(day)}</span>'
            f'<strong>{fmt_int(total)}</strong>'
            '</span>'
            f'{bar(total, max_daily_total)}'
            '<span class="day-card-meta">'
            f'<span>{fmt_int(bucket["requests"])} req</span>'
            f'<span>{fmt_int(avg)} avg</span>'
            f'<span>{fmt_float(prefill)} / {fmt_float(decode)} tok/s</span>'
            f'<span>peak {html.escape(peak_hour[-5:])} ({fmt_int(peak_total)})</span>'
            f'<span>max prompt {fmt_int(bucket["max_prompt"])}</span>'
            '</span>'
            '</button>\n'
        )

with open(hourly_path, 'w') as out:
    for hour in sorted(hourly.keys(), reverse=True):
        bucket = hourly[hour]
        total = bucket['prompt'] + bucket['gen']
        prefill, decode = rates(bucket)
        avg = total / bucket['requests'] if bucket['requests'] else 0.0
        day = hour[:10]
        out.write(
            f'<div class="hour-row" data-day="{html.escape(day)}">'
            '<div class="hour-row-main">'
            f'<span>{html.escape(hour)}</span>'
            f'<strong>{fmt_int(total)}</strong>'
            '</div>'
            f'{bar(total, max_hourly_total)}'
            '<div class="hour-row-meta">'
            f'<span>{fmt_int(bucket["requests"])} req</span>'
            f'<span>{fmt_int(avg)} avg</span>'
            f'<span>{fmt_int(bucket["prompt"])} prompt</span>'
            f'<span>{fmt_int(bucket["gen"])} gen</span>'
            f'<span>{fmt_float(prefill)} / {fmt_float(decode)} tok/s</span>'
            f'<span>{fmt_int(bucket["max_prompt"])} max prompt</span>'
            '</div>'
            '</div>\n'
        )

all_prompt = sum(rec.get('prompt', 0) for rec in complete)
all_gen = sum(rec.get('gen', 0) for rec in complete)
all_prompt_ms = sum(rec.get('prompt_ms', 0.0) for rec in complete)
all_gen_ms = sum(rec.get('gen_ms', 0.0) for rec in complete)
prefill_tps = all_prompt / (all_prompt_ms / 1000) if all_prompt_ms else 0.0
decode_tps = all_gen / (all_gen_ms / 1000) if all_gen_ms else 0.0
total_tokens = all_prompt + all_gen
request_count = len(complete)
avg_total = total_tokens / request_count if request_count else 0.0
max_prompt = max((rec.get('prompt', 0) for rec in complete), default=0)
max_total = max((rec.get('prompt', 0) + rec.get('gen', 0) for rec in complete), default=0)

with open(stats_path, 'w') as out:
    out.write(f'REQUESTS={request_count}\n')
    out.write(f'REQUESTS_FMT={fmt_int(request_count)}\n')
    out.write(f'PROMPT={all_prompt}\n')
    out.write(f'PROMPT_FMT={fmt_int(all_prompt)}\n')
    out.write(f'GENERATED={all_gen}\n')
    out.write(f'GENERATED_FMT={fmt_int(all_gen)}\n')
    out.write(f'TOTAL={total_tokens}\n')
    out.write(f'TOTAL_FMT={fmt_int(total_tokens)}\n')
    out.write(f'PREFILL_TPS={prefill_tps:.1f}\n')
    out.write(f'DECODE_TPS={decode_tps:.1f}\n')
    out.write(f'AVG_TOTAL={avg_total:.0f}\n')
    out.write(f'AVG_TOTAL_FMT={fmt_int(avg_total)}\n')
    out.write(f'MAX_PROMPT={max_prompt}\n')
    out.write(f'MAX_PROMPT_FMT={fmt_int(max_prompt)}\n')
    out.write(f'MAX_TOTAL={max_total}\n')
    out.write(f'MAX_TOTAL_FMT={fmt_int(max_total)}\n')
    out.write(f'DAYS={len(daily)}\n')
    out.write(f'HOURS={len(hourly)}\n')
PY

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
      --bg: #0b0d0f;
      --panel: #15181b;
      --panel-2: #1a1e22;
      --text: #edf0f3;
      --muted: #98a1aa;
      --line: #2a3036;
      --accent: #7fd4b8;
      --accent-2: #9dc6ff;
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
      width: min(1120px, calc(100vw - 28px));
      margin: 18px auto 36px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      margin-bottom: 12px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }
    .sub {
      color: var(--muted);
      margin-top: 3px;
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
    .metrics-strip {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      overflow: hidden;
      margin-bottom: 10px;
    }
    .metric {
      min-height: 72px;
      padding: 12px 14px;
      border-right: 1px solid var(--line);
    }
    .metric:nth-child(4n) { border-right: 0; }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric strong {
      display: block;
      font-size: 22px;
      letter-spacing: 0;
    }
    .secondary-metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .secondary-metrics span {
      border: 1px solid var(--line);
      background: #101316;
      border-radius: 999px;
      padding: 5px 9px;
    }
    section {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      margin-top: 10px;
      overflow: hidden;
    }
    section h2 {
      margin: 0;
      padding: 10px 12px;
      font-size: 15px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
      letter-spacing: 0;
    }
    .section-intro {
      color: var(--muted);
      margin: 0;
      padding: 10px 12px 0;
      font-size: 13px;
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
    .nowrap { white-space: nowrap; }
    .muted { color: var(--muted); }
    .timeline {
      display: grid;
      gap: 8px;
      padding: 10px;
    }
    .day-card {
      width: 100%;
      display: grid;
      grid-template-columns: minmax(150px, 0.55fr) minmax(180px, 1fr) minmax(340px, 1.3fr);
      gap: 12px;
      align-items: center;
      text-align: left;
      color: var(--text);
      border: 1px solid var(--line);
      background: #111519;
      border-radius: 7px;
      padding: 10px 11px;
      cursor: pointer;
      font: inherit;
    }
    .day-card:hover {
      border-color: #45615d;
      background: #13191c;
    }
    .day-card-top {
      display: grid;
      gap: 2px;
    }
    .day-label {
      color: var(--muted);
      font-size: 12px;
    }
    .day-card strong {
      font-size: 18px;
      letter-spacing: 0;
    }
    .day-card-meta,
    .hour-row-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .usage-bar {
      display: block;
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #0b1013;
      border: 1px solid #202a2d;
    }
    .usage-bar span {
      display: block;
      height: 100%;
      background: linear-gradient(90deg, #2d9278, #82d4b8);
    }
    .filters {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #101419;
    }
    .filter,
    .link-button {
      color: var(--text);
      border: 1px solid var(--line);
      background: #11161b;
      border-radius: 6px;
      cursor: pointer;
      font: inherit;
    }
    .filter {
      min-height: 30px;
      padding: 4px 9px;
    }
    .filter.active {
      border-color: #38677a;
      background: #132a35;
      color: #dff7ff;
    }
    .link-button {
      padding: 0;
      border: 0;
      color: var(--accent);
      background: transparent;
      text-decoration: underline;
      text-underline-offset: 3px;
    }
    .hour-list {
      display: grid;
      gap: 6px;
      padding: 10px;
      max-height: 520px;
      overflow: auto;
    }
    .hour-row {
      display: grid;
      grid-template-columns: minmax(132px, 0.45fr) minmax(160px, 1fr) minmax(300px, 1.5fr);
      gap: 10px;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding: 6px 2px 8px;
    }
    .hour-row:last-child {
      border-bottom: 0;
    }
    .hour-row-main {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
      font-variant-numeric: tabular-nums;
    }
    .hour-row-main span {
      color: var(--muted);
      font-size: 12px;
    }
    details {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      margin-top: 10px;
      overflow: hidden;
    }
    summary {
      cursor: pointer;
      padding: 10px 12px;
      background: var(--panel-2);
      color: var(--text);
      font-weight: 600;
      border-bottom: 1px solid transparent;
    }
    details[open] summary {
      border-bottom-color: var(--line);
    }
    .table-wrap {
      overflow-x: auto;
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
      .metrics-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric:nth-child(2n) { border-right: 0; }
      .day-card,
      .hour-row { grid-template-columns: 1fr; }
      .metric strong { font-size: 20px; }
      th, td { padding: 8px 10px; }
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

    <div class="metrics-strip">
      <div class="metric"><span>Completed Requests</span><strong>$REQUESTS_FMT</strong></div>
      <div class="metric"><span>Total Tokens</span><strong>$TOTAL_FMT</strong></div>
      <div class="metric"><span>Prompt Tokens</span><strong>$PROMPT_FMT</strong></div>
      <div class="metric"><span>Generated Tokens</span><strong>$GENERATED_FMT</strong></div>
    </div>
    <div class="secondary-metrics">
      <span>Prefill $PREFILL_TPS tok/s</span>
      <span>Decode $DECODE_TPS tok/s</span>
      <span>Avg $AVG_TOTAL_FMT tokens/request</span>
      <span>Max prompt $MAX_PROMPT_FMT</span>
    </div>

    <section>
      <h2>Daily Usage</h2>
      <p class="section-intro">Click a day to filter the hourly view. Bars compare total tokens across days.</p>
      <div class="timeline">
$(cat "$daily_rows")
      </div>
    </section>

    <section>
      <h2>Hourly Drilldown</h2>
      <div class="filters">
$(cat "$day_filters")
      </div>
      <div id="hourly-table" class="hour-list">
$(cat "$hourly_rows")
      </div>
    </section>

    <details>
      <summary>Recent Requests</summary>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Time</th><th>Task</th><th class="num">Prompt</th><th class="num">Generated</th><th class="num">Total</th><th class="num">Prefill tok/s</th><th class="num">Decode tok/s</th><th class="num">Seconds</th></tr>
          </thead>
          <tbody>
$(cat "$rows")
          </tbody>
        </table>
      </div>
    </details>

    <details>
      <summary>Recent Logs</summary>
      <pre>$(cat "$recent_log")</pre>
    </details>
  </main>
  <script>
    const buttons = Array.from(document.querySelectorAll('.filter'));
    const hourlyRows = Array.from(document.querySelectorAll('#hourly-table .hour-row'));

    function setDay(day) {
      buttons.forEach((button) => button.classList.toggle('active', button.dataset.day === day));
      hourlyRows.forEach((row) => {
        row.hidden = day !== 'all' && row.dataset.day !== day;
      });
    }

    buttons.forEach((button) => {
      button.addEventListener('click', () => setDay(button.dataset.day));
    });

    document.querySelectorAll('[data-day-jump]').forEach((button) => {
      button.addEventListener('click', () => {
        setDay(button.dataset.dayJump);
        document.querySelector('#hourly-table').scrollIntoView({ block: 'start', behavior: 'smooth' });
      });
    });
  </script>
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
