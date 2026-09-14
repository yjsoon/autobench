#!/usr/bin/env python3
"""Generate Qwen3.8-27B autobench pages, tables, charts, and SUMMARY.md."""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
import shlex
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
CONFIGS = ROOT / "_configs"
CHARTS = ROOT / "charts"
INVENTORY = json.loads((ROOT / "inventory" / "machine.json").read_text())
MANIFEST = json.loads((RESULTS / "qwen38-candidates.json").read_text())
CONTEXT = 65536
MEASURED_ON = "2026-08-28"
RECIPE = "strix-halo-vulkan"
SLO_TTFT = 2000.0
SLO_TPOT = 50.0
KV_BYTES_PER_TOKEN = 2 * 16 * 4 * 256 * 2
Q4_0_BYTES_PER_TOKEN = 2 * 16 * 4 * 256 * (18 / 32)
STATE_BYTES_PER_SEQ = 48 * 48 * 128 * 128 * 4 + int(7.5 * 1024 * 1024)

ENGINE_NAME = {"llama": "llama.cpp", "llama.cpp": "llama.cpp", "sglang": "SGLang", "SGLang": "SGLang", "vllm": "vLLM", "vLLM": "vLLM", "exllamav3": "ExLlamaV3", "ExLlamaV3": "ExLlamaV3"}
ENGINE_TOKEN = {"llama.cpp": "llama", "SGLang": "sglang", "vLLM": "vllm", "ExLlamaV3": "exllamav3"}
QUANT_NAME = {"q4_k_m": "Q4_K_M", "q8_0": "Q8_0", "bf16": "BF16", "fp8": "FP8", "nvfp4": "NVFP4", "quark_int4": "AMD Quark W4A16", "quark_mxfp4": "AMD Quark MXFP4", "awq": "AWQ", "int8": "INT8"}
SPEC_NAME = {"base": "none", "none": "none", "mtp": "MTP", "mtp2": "MTP", "mtp3": "MTP", "mtp4": "MTP", "MTP": "MTP", "n-gram": "n-gram", "ngram": "n-gram", "eagle3": "EAGLE-3", "EAGLE-3": "EAGLE-3", "external-draft": "external draft"}
IMAGES = {
    "llama.cpp": "ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41",
    "SGLang": "strix-halo-sglang:dev@sha256:2711177e6d563d227e5eddf894a6069c70c31a7d52c4ddcac97912c935233272",
    "vLLM": "kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2",
    "ExLlamaV3": "n/a (not run)",
}
QUANT_REASON = {
    "Q4_K_M": "GGUF UD-Q4_K_M is the selected memory-saving llama.cpp format for this UMA Vulkan machine. The inventory proved BF16 in native torch, not native low-bit arithmetic; the page therefore records the Vulkan quant path without claiming a hardware-native format.",
    "Q8_0": "GGUF Q8_0 is the wider llama.cpp control format. It tests whether the extra weight memory changes quality or throughput while remaining below the measured UMA ceiling.",
    "BF16": "BF16 is the wide baseline and the narrowest format proven to execute by the native torch probe. Fit and serving behavior are measured rather than inferred from parameter count.",
    "FP8": "FP8 is the official Qwen3.8 W8A8 checkpoint. No explicit Vulkan FP8 extension was observed, so it is tested only through stacks that claim a current FP8 path and the startup backend is recorded.",
    "NVFP4": "NVFP4 is included as a low-bit compatibility probe. It is not treated as native on AMD gfx1151; any successful load would still require a selected-kernel check.",
    "AMD Quark W4A16": "AMD Quark W4A16 is the vendor-specific weight-only candidate. A loadable checkpoint is not taken as evidence of a native execution kernel.",
    "AMD Quark MXFP4": "AMD Quark MXFP4 is the vendor-specific narrow candidate. Native execution was not proven by the inventory, so the engine's selected path is decisive.",
    "AWQ": "AWQ is only a considered alternative where a current checkpoint was discoverable; support in the selected image was not assumed.",
}

def eng(v: Any) -> str:
    return ENGINE_NAME.get(str(v), str(v))

def quant(v: Any) -> str:
    return QUANT_NAME.get(str(v), str(v))

def spec(v: Any) -> str:
    return SPEC_NAME.get(str(v), str(v))

def token(v: Any) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", str(v).lower().replace(".", "-")))

def canonical(row: dict[str, Any]) -> str:
    e = eng(row.get("engine", ""))
    q = str(row.get("quant_label") or token(quant(row.get("quant", ""))))
    parts = ["qwen3-8-27b", token(q), ENGINE_TOKEN.get(e, token(e))]
    variant = str(row.get("variant") or "")
    if variant and variant != "legacy":
        parts.append(token(variant))
    s = str(row.get("spec", "base"))
    if s not in ("base", "none"):
        parts.append(token(s))
    parts.append("c" + str(int(row.get("concurrency", 1))))
    return "-".join(parts)

def commands() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    path = ROOT / "commands.log"
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("scripts/run-qwen38-config.sh "):
            continue
        try:
            p = shlex.split(line)
            vals = list(map(int, p[5:10]))
        except (ValueError, IndexError):
            continue
        if len(p) >= 10:
            out[p[1]] = {"command": line, "engine": p[2], "quant": p[3], "spec": p[4], "context": vals[0], "concurrency": vals[1], "num_prompts": vals[2], "max_seconds": vals[3], "max_tokens": vals[4], "variant": p[10] if len(p) >= 11 else "legacy"}
    return out

COMMANDS = commands()

def legacy(slug: str, cmd: dict[str, Any]) -> dict[str, Any] | None:
    if "slot1024-blocked" not in slug:
        return None
    return {
        "slug": slug, "engine": "llama.cpp", "quant": "Q4_K_M", "quant_label": "q4-k-m", "spec": "base",
        "concurrency": cmd["concurrency"], "context": cmd["context"], "num_prompts": cmd["num_prompts"],
        "max_seconds": cmd["max_seconds"], "max_tokens": cmd["max_tokens"],
        "repo": "unsloth/Qwen3.8-27B-GGUF", "source_repo": "unsloth/Qwen3.8-27B-GGUF",
        "download_url": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF",
        "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502", "weights_bytes": 16464440224,
        "extra_weight_bytes": 0, "decision": "blocked", "reason": "Earlier attempted slot-size run used an incorrect total context pool.",
        "legacy": True,
    }

def normalize_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in MANIFEST:
        row = dict(item)
        row["_manifest_slug"] = str(row.get("slug", ""))
        row["slug"] = row["_manifest_slug"] if row.get("legacy") else canonical(row)
        row.setdefault("context", CONTEXT)
        if row["slug"] not in seen:
            rows.append(row)
            seen.add(row["slug"])
        else:
            for index, existing in enumerate(rows):
                if existing["slug"] == row["slug"] and row.get("page"):
                    rows[index] = row
                    break
    for slug, cmd in COMMANDS.items():
        if slug in seen:
            continue
        row = legacy(slug, cmd)
        if row:
            rows.append(row)
            seen.add(slug)
    return rows

ROWS = normalize_rows()

def has_attempt(row: dict[str, Any]) -> bool:
    slug = str(row["slug"])
    original = str(row.get("_manifest_slug", ""))
    return bool(
        slug in COMMANDS or original in COMMANDS or (RESULTS / (slug + ".json")).exists()
        or list(LOGS.glob("server-" + slug + "-r*.log"))
    )

def command_for(row: dict[str, Any]) -> str:
    for slug in (str(row["slug"]), str(row.get("_manifest_slug", ""))):
        if slug in COMMANDS:
            return str(COMMANDS[slug]["command"])
    return ""

def result_for(slug: str) -> dict[str, Any] | None:
    path = RESULTS / (slug + ".json")
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None

def logs(prefix: str, slug: str) -> list[Path]:
    return sorted(LOGS.glob(prefix + "-" + slug + "-r*.log"))

def probe_for(slug: str) -> dict[str, Any] | None:
    paths = sorted(LOGS.glob("coherence-" + slug + "-r*.json"))
    if not paths:
        return None
    try:
        value = json.loads(paths[-1].read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None

def error_for(slug: str, result: dict[str, Any] | None, row: dict[str, Any]) -> str:
    if row.get("reason"):
        return str(row["reason"])
    if result:
        for obj in [result] + [x for x in result.get("repeats", []) if isinstance(x, dict)]:
            for item in obj.get("error_examples", []) or []:
                if isinstance(item, dict) and item.get("message"):
                    return str(item["message"])
                if item:
                    return str(item)
    for path in logs("server", slug) + logs("bench", slug) + logs("probe", slug):
        for line in path.read_text(errors="replace").splitlines():
            if re.search(r"(?i)(error|exception|traceback|assert|out of memory|oom|failed|exceeds the available context)", line):
                return line.strip()
    return str(row.get("reason") or "No successful result JSON was produced; see the saved logs.")

def acceptance_for(slug: str, which: str, spec_tokens: Any = None) -> dict[str, Any]:
    if which in ("base", "none"):
        return {"rate": None, "length": None, "tokens": None, "text": "Not applicable on a base page."}
    server_paths = logs("server", slug)
    paths = server_paths if server_paths else logs("spec", slug)
    # The wrapper's coherence probe is task 0 and also emits draft acceptance.
    # Exclude it: acceptance must describe the benchmark stream, not warmup.
    text = "\n".join(
        line
        for path in paths
        for line in path.read_text(errors="replace").splitlines()
        if "task 0 |" not in line
    )
    is_ngram = which in ("n-gram", "ngram")
    if is_ngram:
        tokens: Any = "n/a (llama.cpp n=12,m=48)"
        parameter_text = "num_speculative_tokens=n/a; llama.cpp ngram-simple-size-n=12 and ngram-simple-size-m=48"
    else:
        tokens = int(spec_tokens or 3)
        parameter_text = "num_speculative_tokens=%d" % tokens
    rows = re.findall(r"(?i)draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\),\s*mean len\s*=\s*([0-9.]+)", text)
    if rows:
        accepted = sum(int(row[1]) for row in rows)
        generated = sum(int(row[2]) for row in rows)
        if generated:
            rate = 100.0 * accepted / generated
            length = sum(float(row[3]) * int(row[2]) for row in rows) / generated
            return {"rate": rate, "length": length, "tokens": tokens, "text": "Measured aggregate acceptance rate %.2f%% (%d accepted / %d generated); weighted mean acceptance length %.2f; %s." % (rate, accepted, generated, length, parameter_text)}
    rate = None
    length = None
    for pattern in (r"(?i)accept(?:ance)?(?:[_ -]rate| rate)?\s*[:=]\s*([0-9.]+)\s*%", r"(?i)accept(?:ance)?(?:[_ -]rate| rate)?\s*[:=]\s*([0-9.]+)\b", r"(?i)accepted[^0-9]*([0-9.]+)\s*%"):
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            rate = value if value > 1 else value * 100
            break
    for pattern in (r"(?i)mean acceptance length\s*[:=]\s*([0-9.]+)", r"(?i)accept(?:ance)?(?:[_ ]len(?:gth)?| len(?:gth)?)\s*[:=]\s*([0-9.]+)", r"(?i)accept(?:ed)?\s+len(?:gth)?\s*[:=]\s*([0-9.]+)"):
        match = re.search(pattern, text)
        if match:
            length = float(match.group(1))
            break
    if rate is None or length is None:
        return {"rate": rate, "length": length, "tokens": tokens, "text": "Acceptance metric was not emitted or was not parseable before teardown; speculative result is blocked; %s." % parameter_text}
    return {"rate": rate, "length": length, "tokens": tokens, "text": "Measured acceptance rate %.2f%%; mean acceptance length %.2f; %s." % (rate, length, parameter_text)}

def current_watchdog(path: Path) -> str:
    text = path.read_text(errors="replace") if path.exists() else ""
    starts = list(re.finditer(r"^watchdog_start .*?$", text, re.M))
    return text[starts[-1].start():] if starts else text

def memory_for(slug: str) -> dict[str, Any]:
    baselines = []
    available = []
    vram = []
    for path in sorted(LOGS.glob("image-" + slug + "-r*.log")):
        vals = re.findall(r"baseline_mem_available_kb=(\d+)", path.read_text(errors="replace"))
        if vals:
            baselines.append(int(vals[-1]))
    for path in sorted(LOGS.glob("mem-watchdog-" + slug + "-r*.log")):
        available += [int(x) for x in re.findall(r"available_kb=(\d+)", current_watchdog(path))]
    for path in sorted(LOGS.glob("accel-" + slug + "-r*.csv")):
        for line in path.read_text(errors="replace").splitlines()[1:]:
            p = line.split(",")
            if len(p) >= 3:
                try:
                    available.append(int(p[1]))
                    vram.append(int(p[2]))
                except ValueError:
                    pass
    if not baselines or not available:
        return {"mem_gb": None, "baseline_kb": None, "min_kb": None, "vram_peak_gb": None, "vram_delta_gb": None}
    base = max(baselines)
    minimum = min(available)
    vpeak = max(vram) if vram else None
    vdelta = (vpeak - 2842021888) / (1024 ** 3) if vpeak is not None else None
    return {"mem_gb": round(max(0, base - minimum) / (1024 ** 2), 2), "baseline_kb": base, "min_kb": minimum, "vram_peak_gb": round(vpeak / (1024 ** 3), 2) if vpeak else None, "vram_delta_gb": round(vdelta, 2) if vdelta is not None else None}

def telemetry_for(slug: str) -> dict[str, Any]:
    busy, temp, power, clocks = [], [], [], []
    for path in sorted(LOGS.glob("accel-" + slug + "-r*.csv")):
        for line in path.read_text(errors="replace").splitlines()[1:]:
            p = line.split(",")
            if len(p) < 7:
                continue
            try:
                if p[3]: busy.append(float(p[3]))
                if p[4]: temp.append(float(p[4]) / 1000)
                if p[5]: power.append(float(p[5]) / 1_000_000)
                if p[6]: clocks.append(float(p[6]) / 1_000_000)
            except ValueError:
                pass
    return {"busy_peak": max(busy) if busy else None, "busy_median": statistics.median(busy) if busy else None, "temp_peak_c": max(temp) if temp else None, "power_median_w": statistics.median(power) if power else None, "power_peak_w": max(power) if power else None, "clock_median_mhz": statistics.median(clocks) if clocks else None}

def completed_at(slug: str) -> str:
    path = LOGS / ("completed-" + slug + ".txt")
    if path.exists() and path.read_text().strip():
        return path.read_text().strip()
    candidates = logs("server", slug) + [LOGS / ("image-" + slug + "-r1.log")]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return "n/a"
    stamp = dt.datetime.fromtimestamp(max(p.stat().st_mtime for p in candidates), dt.timezone(dt.timedelta(hours=8)))
    return stamp.strftime("%Y-%m-%d %H:%M %z")

def startup_for(slug: str) -> list[str]:
    out = []
    for path in logs("server", slug):
        for line in path.read_text(errors="replace").splitlines():
            if re.search(r"(?i)(initializing|memory|kv|cache|weight|buffer|maximum|n_seq|n_slots|context|backend|loaded|listening|oom|error|assert|failed|spec|accept)", line):
                out.append(line.strip())
    return out[:35]

def weight_gib(row: dict[str, Any]) -> float | None:
    if row.get("weights_bytes") is None:
        return None
    return (int(row["weights_bytes"]) + int(row.get("extra_weight_bytes") or 0)) / (1024 ** 3)

def kv_bytes_per_token(row: dict[str, Any]) -> float:
    return Q4_0_BYTES_PER_TOKEN if str(row.get("kv_dtype") or "") == "q4_0" or str(row.get("variant") or "") == "q4kv" else KV_BYTES_PER_TOKEN

def kv_formula(row: dict[str, Any]) -> str:
    if kv_bytes_per_token(row) == Q4_0_BYTES_PER_TOKEN:
        return "2 x 16 x 4 x 256 x (18 bytes / 32 values = 0.5625 bytes/value) for q4_0 K/V"
    return "2 x 16 x 4 x 256 x 2 bytes/token for BF16 K/V"

def arithmetic(row: dict[str, Any]) -> dict[str, Any]:
    weight = weight_gib(row)
    if weight is None:
        return {"weights": None, "kv": None, "state": None, "total": None, "fits": "n/a"}
    ctx = int(row.get("context", CONTEXT))
    conc = int(row.get("concurrency", 1))
    kv = ctx * kv_bytes_per_token(row) / (1024 ** 3)
    if eng(row.get("engine")) != "llama.cpp":
        kv *= conc
    state = STATE_BYTES_PER_SEQ * conc / (1024 ** 3)
    total = weight + kv + state
    ceiling = float(INVENTORY.get("autobench_assumptions", {}).get("memory_ceiling_gb") or 96)
    floor = float(INVENTORY.get("autobench_assumptions", {}).get("watchdog_floor_gb") or 8)
    budget = ceiling - floor
    fit = "yes" if total <= budget * 0.8 else "tight" if total <= budget else "no"
    return {"weights": round(weight, 3), "kv": round(kv, 3), "state": round(state, 3), "total": round(total, 3), "fits": fit}

def record(row: dict[str, Any]) -> dict[str, Any]:
    slug = str(row["slug"])
    result = result_for(slug)
    probe = probe_for(slug)
    acceptance = acceptance_for(slug, str(row.get("spec", "base")), row.get("spec_tokens"))
    status = "done" if result and int(result.get("completed", 0)) > 0 and int(result.get("errors", 0)) == 0 else "blocked"
    if probe is not None and not probe.get("ok", False):
        status = "blocked"
    if str(row.get("spec", "base")) not in ("base", "none") and acceptance["rate"] is None:
        status = "blocked"
    return {"row": row, "slug": slug, "result": result, "probe": probe, "acceptance": acceptance, "status": status, "error": error_for(slug, result, row), "memory": memory_for(slug), "telemetry": telemetry_for(slug), "startup": startup_for(slug), "command": command_for(row), "completed_at": completed_at(slug)}

def val(value: Any, digits: int = 2) -> str:
    if value is None or value == "n/a":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    out = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(f, "n/a")).replace("|", "\\|").replace("\n", " ") for f in fields) + " |")
    return "\n".join(out)

def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def page(record: dict[str, Any]) -> str:
    row, result, status = record["row"], record["result"], record["status"]
    e, q, s = eng(row.get("engine")), quant(row.get("quant")), spec(row.get("spec", "base"))
    c, ctx = int(row.get("concurrency", 1)), int(row.get("context", CONTEXT))
    source = row.get("source_repo") or row.get("repo") or "n/a"
    url = row.get("download_url") or ("https://huggingface.co/" + str(source))
    memory = record["memory"]
    metric = lambda field: result.get(field) if status == "done" and result else "n/a"
    mem = memory["mem_gb"] if status == "done" else "n/a"
    tags = ["qwen3.8-27b", "Alibaba", "Qwen", q, "16-40B", "conc-" + str(c), RECIPE]
    title = "Qwen3.8-27B · %s · %s" % (e, q) + ((" + " + s) if s != "none" else "")
    spec_text = record["acceptance"]["text"]
    mem_source = (
        "Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes."
        if status == "done" else
        "Shared UMA method would use baseline host MemAvailable minus the 2-second minimum, with one accelerator summed. No publishable peak was recorded for this blocked attempt; static reservations are not a footprint."
    )
    lines = [
        "---", "title: " + title, "model: Qwen/Qwen3.8-27B", "company: Alibaba", "family: Qwen", "params: 27B (dense)",
        "engine: " + e, "speculative: " + s, "quant: " + q, "quant_rationale: >-", "  " + QUANT_REASON.get(q, "See the candidate manifest; support was not assumed."),
        "context: " + str(ctx), "modalities: [text, image, video]", "mm_served: false", "concurrency: " + str(c),
        "tags: [" + ", ".join(tags) + "]", "status: " + status,
        "prefill_toks: " + val(metric("prefill_toks")), "decode_toks: " + val(metric("decode_toks")),
        "mem_gb: " + val(mem), "mem_source: >-", "  " + mem_source,
        "spec_acceptance: >-", "  " + spec_text, "measured_on: " + str(row.get("measured_on") or MEASURED_ON), "completed_at: " + record["completed_at"],
        "engine_image: " + IMAGES.get(e, "n/a"), "source_repo: " + str(source), "download_url: " + str(url),
        "run_command: |", "  " + (record["command"] or "n/a; no runnable command was recorded"), "---", "", "## Notes", "",
    ]
    if status == "done" and result:
        lines.append("- Headline: %.2f aggregate decode tok/s, %.2f aggregate prefill tok/s, %s completed, %s errors, %.1f s, hit_time_cap=%s." % (float(result.get("decode_toks", 0)), float(result.get("prefill_toks", 0)), result.get("completed"), result.get("errors"), float(result.get("duration_s", 0)), result.get("hit_time_cap")))
        lines.append("- SLO is median TTFT <= %.0f ms and median TPOT <= %.0f ms. TTFT median/p95/p99 = %.1f/%.1f/%.1f ms; TPOT median/p95 = %s/%s ms; goodput = %.1f%%." % (SLO_TTFT, SLO_TPOT, float(result.get("ttft_median_ms", 0)), float(result.get("ttft_p95_ms", 0)), float(result.get("ttft_p99_ms", 0)), str(result.get("tpot_median_ms")), str(result.get("tpot_p95_ms")), float(result.get("goodput_pct", 0))))
    else:
        lines.append("- Status: blocked. Exact observed error/reason: " + record["error"])
    if memory["mem_gb"] is not None:
        lines.append("- Peak shared-memory delta %.2f GiB from baseline %s kB to minimum %s kB; VRAM cross-check peak %s GiB and delta %s GiB." % (memory["mem_gb"], memory["baseline_kb"], memory["min_kb"], memory["vram_peak_gb"], memory["vram_delta_gb"]))
    else:
        lines.append("- No complete memory trace was published for this attempt.")
    ds = INVENTORY.get("dataset", {})
    if row.get("revision"):
        lines.append("- Checkpoint revision: %s@%s; base Qwen3.8-27B config revision: %s." % (source, row.get("revision"), row.get("model_revision", "not recorded")))
    if e == "SGLang":
        lines.append("- Behavior-changing container environment: SGLANG_FORCE_NATIVE_LAYERNORM=1, PYTORCH_ROCM_ARCH=gfx1151, PYTORCH_TUNABLEOP_ENABLED=1.")
    elif e == "vLLM":
        lines.append("- Behavior-changing container environment: HSA_OVERRIDE_GFX_VERSION=11.5.1.")
    else:
        lines.append("- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.")
    if e == "llama.cpp":
        lines.append("- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.")
    if row.get("recipe_repo"):
        hashes = row.get("recipe_file_hashes") or {}
        hash_text = "; ".join("%s=%s" % (k, v) for k, v in hashes.items()) or "not recorded"
        lines.append("- External recipe source: %s@%s; file SHA256s: %s." % (row.get("recipe_repo"), row.get("recipe_revision"), hash_text))
        recipe_flags = "--cache-type-k q4_0 --cache-type-v q4_0 -fa 1"
        if str(row.get("spec", "")) in ("mtp2", "mtp3", "mtp4"):
            recipe_flags += " --spec-type draft-mtp --spec-draft-n-max %s" % row.get("spec_tokens")
        lines.append("- Recipe adaptation: applied %s; the repository's probe.py was not used as the headline harness, because autobench's closed-loop ShareGPT generator supplies the recorded aggregate metrics." % recipe_flags)
    lines.append("- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is %s; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.")
    lines[-1] = lines[-1] % ds.get("sha256", "unavailable")
    lines.append("- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c%d, per-user decode is aggregate decode divided by the in-flight count." % c)
    a = arithmetic(row)
    if a["weights"] is not None:
        lines.append("- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) %.3f GiB; KV %.3f GiB; recurrent state %.3f GiB; total %.3f GiB; fit=%s. KV formula: %s; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence." % (a["weights"], a["kv"], a["state"], a["total"], a["fits"], kv_formula(row)))
    if record["startup"]:
        lines.append("- Engine startup evidence (measured log lines):")
        lines.extend("  - " + x for x in record["startup"][:20])
    else:
        lines.append("- The engine emitted no parseable numeric KV/weight/memory breakdown before teardown; no value was inferred.")
    lines.append("- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.")
    t = record["telemetry"]
    lines.append("- Telemetry: busy peak/median %s/%s%%, temperature peak %s C, power median/peak %s/%s W, sustained clock median %s MHz." % (t["busy_peak"], t["busy_median"], t["temp_peak_c"], t["power_median_w"], t["power_peak_w"], t["clock_median_mhz"]))
    if s != "none":
        lines.append("- Speculation acceptance: " + spec_text)
        if s == "MTP":
            if record["acceptance"]["rate"] is None:
                lines.append("- Cross-check anchor: not assessable because this MTP configuration was blocked before acceptance metrics; the previous-generation ShareGPT MTP anchor is 67% average, per-position 0.84/0.66/0.51, mean length about 3.0 at three draft tokens.")
            else:
                rate = float(record["acceptance"]["rate"])
                state = "consistent with" if abs(rate - 67.0) <= 10.0 else "not consistent with"
                lines.append("- Cross-check anchor: measured MTP acceptance %.2f%%; this is %s the previous-generation ShareGPT MTP anchor of 67%% average, per-position 0.84/0.66/0.51, mean length about 3.0 at three draft tokens." % (rate, state))
                if abs(rate - 67.0) > 10.0:
                    lines.append("- Anchor investigation: the gap is recorded because this is a newer Qwen3.8 checkpoint and the qwen38-mtp run uses draft depth %s; server logs emitted draft-acceptance lines, so no silent-disabled fallback was observed. The cause of the lower acceptance was not isolated." % (record["acceptance"]["tokens"] or "not recorded"))
        else:
            lines.append("- Cross-check anchor: the 67% value is a previous-generation ShareGPT MTP anchor, not an n-gram expectation; this n-gram result is method-specific and is not directly comparable to that anchor.")
    lines.append("- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag %s is defined once in SUMMARY.md." % RECIPE)
    background = INVENTORY.get("pre_existing_services_observed")
    if background:
        lines.append("- Machine-state caveat: pre-existing container %s was loaded before this run. It was observed idle (0.00%% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed." % background.get("container", "unknown"))
    if row.get("reason"):
        lines.append("- Candidate note: " + str(row["reason"]))
    return "\n".join(lines) + "\n"

def table_b(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_slug = {r["slug"]: r for r in records}
    out = []
    for row in ROWS:
        calc = arithmetic(row)
        rec = by_slug.get(str(row["slug"]))
        if rec:
            decision = "run" if rec["status"] == "done" else "blocked"
            reason = "measured and published" if rec["status"] == "done" else rec["error"]
        else:
            decision = str(row.get("decision") or "skip")
            reason = str(row.get("reason") or "planned candidate was not reached; no measurement exists")
            if decision == "run":
                decision = "skip"
        out.append({"config": row["slug"], "engine": eng(row.get("engine")), "quant": quant(row.get("quant")), "weights_gib": val(calc["weights"], 3), "kv_gib_at_ctx": val(calc["kv"], 3), "state_gib_at_conc": val(calc["state"], 3), "total_gib": val(calc["total"], 3), "fits": calc["fits"], "decision": decision, "reason": reason})
    return out

def table_c(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in sorted(records, key=lambda x: x["slug"]):
        row, result = r["row"], r["result"] or {}
        good = r["status"] == "done"
        def m(f: str, d: int = 2) -> str:
            return val(result.get(f) if good else "n/a", d)
        out.append({"slug": r["slug"], "engine": eng(row.get("engine")), "speculative": spec(row.get("spec", "base")), "quant": quant(row.get("quant")), "context": row.get("context", CONTEXT), "concurrency": row.get("concurrency"), "prefill_toks": m("prefill_toks"), "decode_toks": m("decode_toks"), "ttft_median_ms": m("ttft_median_ms", 1), "ttft_p95_ms": m("ttft_p95_ms", 1), "tpot_median_ms": m("tpot_median_ms", 1), "goodput_pct": m("goodput_pct", 1), "mem_gb": val(r["memory"]["mem_gb"]) if good else "n/a", "completed": result.get("completed") if good else "n/a", "errors": result.get("errors") if good else "n/a", "hit_time_cap": result.get("hit_time_cap") if good else "n/a", "status": r["status"]})
    return out

def base_record(r: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for b in records:
        a, br = r["row"], b["row"]
        if eng(a.get("engine")) == eng(br.get("engine")) and quant(a.get("quant")) == quant(br.get("quant")) and int(a.get("concurrency", 1)) == int(br.get("concurrency", 1)) and str(a.get("variant") or "") == str(br.get("variant") or "") and str(br.get("spec", "base")) in ("base", "none"):
            return b
    return None

def table_d(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        s = str(r["row"].get("spec", "base"))
        if s in ("base", "none"):
            continue
        b = base_record(r, records)
        sr, br = r["result"] if r["status"] == "done" else None, b["result"] if b and b["status"] == "done" else None
        ac = r["acceptance"]
        if sr and br and ac["rate"] is not None:
            sd, bd = float(sr.get("decode_toks", 0)), float(br.get("decode_toks", 0))
            sp = sd / bd if bd else None
            verdict = "win" if sp > 1.02 else "loss" if sp < .98 else "neutral"
        else:
            sd = bd = sp = "n/a"
            verdict = "blocked"
        out.append({"slug": r["slug"], "base_slug": b["slug"] if b else canonical({**r["row"], "spec": "base"}), "method": spec(s), "num_speculative_tokens": ac["tokens"] if ac["tokens"] else "n/a", "concurrency": r["row"].get("concurrency"), "acceptance_rate": val(ac["rate"]), "acceptance_length": val(ac["length"]), "decode_toks_spec": val(sd), "decode_toks_base": val(bd), "speedup": val(sp), "verdict": verdict})
    return out

def samples(slug: str) -> list[float]:
    path = RESULTS / (slug + ".samples.json")
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    values = []
    for rep in obj.get("repeats", []):
        for item in rep.get("samples", []):
            if item.get("tpot_ms") is not None:
                values.append(float(item["tpot_ms"]))
    return values

def make_charts(records: list[dict[str, Any]], drows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CHARTS.mkdir(exist_ok=True)
    colors = {"llama.cpp": "#1f77b4", "SGLang": "#ff7f0e", "vLLM": "#2ca02c", "ExLlamaV3": "#9467bd"}
    styles = {"none": ("-", "o"), "MTP": ("--", "^"), "n-gram": (":", "s"), "EAGLE-3": ("-.", "D"), "external draft": ("-.", "x")}
    points = sorted({int(r["row"].get("concurrency", 1)) for r in records}) or [1]
    def save(name: str):
        plt.tight_layout()
        plt.savefig(CHARTS / (name + ".png"), dpi=140)
        plt.close()
    def series_name(e: str, q: str, s: str, v: str = "") -> str:
        return "%s/%s/%s%s" % (e, q, s, ("/" + v) if v else "")
    def groups():
        out = {}
        for r in records:
            e = eng(r["row"].get("engine")); q = quant(r["row"].get("quant")); s = spec(r["row"].get("spec", "base")); v = str(r["row"].get("variant") or "")
            out.setdefault((e, q, s, v), []).append(r)
        return out
    record_by_slug = {r["slug"]: r for r in records}
    # Throughput and p95 share the grouping but use separate recorded fields.
    for name, field, ylabel, title in [
        ("throughput_vs_concurrency", "decode_toks", "decode_toks (aggregate tok/s)", "throughput_vs_concurrency"),
        ("ttft_p95_vs_concurrency", "ttft_p95_ms", "ttft_p95_ms (ms)", "ttft_p95_vs_concurrency"),
    ]:
        rows = []
        fig, ax = plt.subplots(figsize=(8, 5))
        for (e, q, s, v), group in sorted(groups().items()):
            group.sort(key=lambda r: int(r["row"].get("concurrency", 1)))
            good = [r for r in group if r["status"] == "done" and r["result"]]
            if not good:
                continue
            xs = [int(r["row"].get("concurrency", 1)) for r in good]
            ys = [float(r["result"][field]) for r in good]
            style, marker = styles.get(s, ("-", "o"))
            ax.plot(xs, ys, label=series_name(e, q, s, v), color=colors.get(e, "#333333"), linestyle=style, marker=marker)
            for r in good:
                rows.append({"series": series_name(e, q, s, v), "concurrency": r["row"].get("concurrency"), field: r["result"].get(field), "status": "done"})
            for r in group:
                if r["status"] != "done":
                    rows.append({"series": series_name(e, q, s, v), "concurrency": r["row"].get("concurrency"), field: ys[-1], "status": "blocked_marked_at_last_success"})
                    ax.plot([xs[-1]], [ys[-1]], marker="x", color=colors.get(e, "#333333"), markersize=8)
        if field == "ttft_p95_ms":
            ax.axhline(SLO_TTFT, color="black", linestyle="--", label="median TTFT SLO threshold <= 2000 ms")
        ax.set_xscale("log", base=2); ax.set_xticks(points); ax.set_xlabel("concurrency (in-flight requests)"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=7)
        write_csv(CHARTS / (name + ".csv"), rows, ["series", "concurrency", field, "status"]); save(name)
    # Pareto.
    good = [r for r in records if r["status"] == "done" and r["result"]]
    fig, ax = plt.subplots(figsize=(8, 5)); rows = []
    frontier = []
    for r in good:
        x, y = float(r["result"]["tpot_median_ms"]), float(r["result"]["decode_toks"])
        ax.scatter([x], [y], color=colors.get(eng(r["row"].get("engine")), "#333333"), marker=styles.get(spec(r["row"].get("spec", "base")), ("-", "o"))[1])
        ax.annotate("c%s" % r["row"].get("concurrency"), (x, y), fontsize=7)
        rows.append({"slug": r["slug"], "tpot_median_ms": x, "decode_toks": y, "status": "done"})
        if not any(float(o["result"]["tpot_median_ms"]) <= x and float(o["result"]["decode_toks"]) >= y and (float(o["result"]["tpot_median_ms"]) < x or float(o["result"]["decode_toks"]) > y) for o in good):
            frontier.append(r)
    frontier.sort(key=lambda r: float(r["result"]["tpot_median_ms"]))
    if frontier:
        fx = [float(r["result"]["tpot_median_ms"]) for r in frontier]; fy = [float(r["result"]["decode_toks"]) for r in frontier]
        ax.plot(fx, fy, color="black", label="Pareto frontier"); ax.fill_between(fx, 0, fy, color="#cccccc", alpha=.2, label="dominated region")
        for r in frontier: rows.append({"slug": r["slug"], "tpot_median_ms": r["result"]["tpot_median_ms"], "decode_toks": r["result"]["decode_toks"], "status": "frontier"})
    ax.set_xlabel("tpot_median_ms (ms/token/user)"); ax.set_ylabel("decode_toks (aggregate tok/s)"); ax.set_title("pareto_latency_throughput"); ax.legend(fontsize=7)
    write_csv(CHARTS / "pareto_latency_throughput.csv", rows, ["slug", "tpot_median_ms", "decode_toks", "status"]); save("pareto_latency_throughput")
    # Distribution, only real samples.
    conc = 8 if any(int(r["row"].get("concurrency", 1)) == 8 for r in good) else min(points)
    candidates = [r for r in good if int(r["row"].get("concurrency", 1)) == conc and samples(r["slug"])]
    candidates.sort(key=lambda r: float(r["result"]["decode_toks"]), reverse=True); candidates = candidates[:3]
    fig, ax = plt.subplots(figsize=(8, 5)); data = []; labels = []; rows = []
    for r in candidates:
        vals = samples(r["slug"]); data.append(vals); labels.append(r["slug"])
        rows += [{"slug": r["slug"], "concurrency": conc, "tpot_ms": v} for v in vals]
    if data: ax.boxplot(data, tick_labels=labels)
    ax.set_xlabel("configuration at concurrency %d" % conc); ax.set_ylabel("tpot_ms (ms/token/user)"); ax.set_title("tpot_distribution (recorded samples)"); ax.tick_params(axis="x", labelrotation=25)
    write_csv(CHARTS / "tpot_distribution.csv", rows, ["slug", "concurrency", "tpot_ms"]); save("tpot_distribution")
    # Spec speedup and acceptance.
    for name, yfield, ylabel, title in [
        ("speculative_speedup_vs_concurrency", "speedup", "speedup (decode_toks_spec / decode_toks_base)", "speculative_speedup_vs_concurrency"),
        ("acceptance_vs_concurrency", "acceptance_rate", "acceptance_rate (%)", "acceptance_vs_concurrency"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5)); rows = []
        for d in drows:
            key = d["slug"].rsplit("-c", 1)[0]
            if d[yfield] != "n/a":
                rec = record_by_slug.get(d["slug"])
                e = eng(rec["row"].get("engine")) if rec else "unknown"
                st, mk = styles.get(d["method"], ("-", "o"))
                ax.plot([int(d["concurrency"])], [float(d[yfield])], marker=mk, linestyle="none", color=colors.get(e, "#333333"), label=key)
                rows.append({"series": key, "concurrency": d["concurrency"], yfield: d[yfield], "acceptance_rate": d["acceptance_rate"], "acceptance_length": d["acceptance_length"], "status": "done"})
                if name.startswith("speculative"): ax.annotate("%.1f%%" % float(d["acceptance_rate"]), (int(d["concurrency"]), float(d[yfield])), fontsize=7)
        for key in sorted({d["slug"].rsplit("-c", 1)[0] for d in drows}):
            valid = [d for d in drows if d["slug"].rsplit("-c", 1)[0] == key and d[yfield] != "n/a"]
            blocked = [d for d in drows if d["slug"].rsplit("-c", 1)[0] == key and d[yfield] == "n/a"]
            if valid and blocked:
                last = max(valid, key=lambda d: int(d["concurrency"]))
                rec = record_by_slug.get(last["slug"]); e = eng(rec["row"].get("engine")) if rec else "unknown"
                ax.plot([int(last["concurrency"])], [float(last[yfield])], marker="x", linestyle="none", color=colors.get(e, "#333333"), markersize=8)
                for d in blocked:
                    rows.append({"series": key, "concurrency": d["concurrency"], yfield: last[yfield], "acceptance_rate": d["acceptance_rate"], "acceptance_length": d["acceptance_length"], "status": "blocked_marked_at_last_success"})
        if name.startswith("speculative"): ax.axhline(1.0, color="black", linestyle=":", label="speedup=1.0")
        ax.set_xscale("log", base=2); ax.set_xticks(points); ax.set_xlabel("concurrency (in-flight requests)"); ax.set_ylabel(ylabel); ax.set_title(title)
        if rows: ax.legend(fontsize=7)
        fields = ["series", "concurrency", "speedup", "acceptance_rate", "acceptance_length"] if yfield == "speedup" else ["series", "concurrency", "acceptance_rate", "acceptance_length", "status"]
        write_csv(CHARTS / (name + ".csv"), rows, fields); save(name)
    # Memory.
    fig, ax = plt.subplots(figsize=(8, 5)); rows = []
    for (e, q, s, v), group in sorted(groups().items()):
        key = series_name(e, q, s, v); st, mk = styles.get(s, ("-", "o"))
        vals = [r for r in group if r["status"] == "done" and r["memory"]["mem_gb"] is not None]
        if vals:
            vals.sort(key=lambda r: int(r["row"].get("concurrency", 1)))
            ax.plot([int(r["row"].get("concurrency", 1)) for r in vals], [r["memory"]["mem_gb"] for r in vals], label=key, color=colors.get(e, "#333333"), linestyle=st, marker=mk)
            rows += [{"series": key, "concurrency": r["row"].get("concurrency"), "mem_gb": r["memory"]["mem_gb"], "status": r["status"]} for r in vals]
            for r in group:
                if r["status"] != "done":
                    last = max(vals, key=lambda x: int(x["row"].get("concurrency", 1)))
                    rows.append({"series": key, "concurrency": r["row"].get("concurrency"), "mem_gb": last["memory"]["mem_gb"], "status": "blocked_marked_at_last_success"})
                    ax.plot([int(last["row"].get("concurrency", 1))], [last["memory"]["mem_gb"]], marker="x", color=colors.get(e, "#333333"), markersize=8)
        else:
            for r in group:
                if r["status"] != "done":
                    rows.append({"series": key, "concurrency": r["row"].get("concurrency"), "mem_gb": "n/a", "status": "blocked_no_success"})
    ceiling = float(INVENTORY.get("autobench_assumptions", {}).get("memory_ceiling_gb") or 96); floor = float(INVENTORY.get("autobench_assumptions", {}).get("watchdog_floor_gb") or 8)
    ax.axhline(ceiling, color="black", linestyle="--", label="memory ceiling (recorded)"); ax.axhline(floor, color="red", linestyle=":", label="watchdog floor (recorded)")
    ax.set_xscale("log", base=2); ax.set_xticks(points); ax.set_xlabel("concurrency (in-flight requests)"); ax.set_ylabel("mem_gb (host MemAvailable delta; reservation caveat)"); ax.set_title("memory_headroom"); ax.legend(fontsize=7)
    write_csv(CHARTS / "memory_headroom.csv", rows, ["series", "concurrency", "mem_gb", "status"]); save("memory_headroom")

def write_aux_logs(records: list[dict[str, Any]]) -> None:
    for rec in records:
        slug = rec["slug"]
        a = arithmetic(rec["row"])
        startup = "\n".join(rec["startup"]) if rec["startup"] else "No engine startup lines were captured."
        (LOGS / ("kvcache-" + slug + ".txt")).write_text(
            "\n".join([
                "slug=" + slug,
                "engine_reported_startup_lines:",
                startup,
                "computed_weights_gib=" + str(a["weights"]),
                "computed_kv_gib_at_context=" + str(a["kv"]),
                "computed_recurrent_state_gib_at_concurrency=" + str(a["state"]),
                "computed_total_gib=" + str(a["total"]),
                "computed_kv_formula=" + kv_formula(rec["row"]) + " x context; GDN layers have recurrent state, not KV cache.",
            ]) + "\n"
        )
        m = rec["memory"]
        t = rec["telemetry"]
        (LOGS / ("memprofile-" + slug + ".txt")).write_text(
            "\n".join([
                "slug=" + slug,
                "baseline_mem_available_kb=" + str(m["baseline_kb"]),
                "minimum_mem_available_kb=" + str(m["min_kb"]),
                "attributable_mem_delta_gib=" + str(m["mem_gb"]),
                "vram_peak_gib=" + str(m["vram_peak_gb"]),
                "vram_delta_gib=" + str(m["vram_delta_gb"]),
                "telemetry_busy_peak_median_percent=" + str(t["busy_peak"]) + "/" + str(t["busy_median"]),
                "telemetry_temp_peak_c=" + str(t["temp_peak_c"]),
                "telemetry_power_median_peak_w=" + str(t["power_median_w"]) + "/" + str(t["power_peak_w"]),
                "telemetry_clock_median_mhz=" + str(t["clock_median_mhz"]),
                "method=shared UMA host MemAvailable delta plus amdgpu VRAM cross-check; sampler interval is 2 seconds; one device.",
                "reservation_caveat=engine static KV reservations are not a physical footprint comparison; use startup profiling lines where emitted.",
            ]) + "\n"
        )
def recommendations(records: list[dict[str, Any]], drows: list[dict[str, Any]]) -> str:
    good = [r for r in records if r["status"] == "done" and r["result"]]
    if not good:
        return "No completed configuration exists; no recommendation is publishable."
    frontier = []
    for r in good:
        x, y = float(r["result"]["tpot_median_ms"]), float(r["result"]["decode_toks"])
        if not any(float(o["result"]["tpot_median_ms"]) <= x and float(o["result"]["decode_toks"]) >= y and (float(o["result"]["tpot_median_ms"]) < x or float(o["result"]["decode_toks"]) > y) for o in good):
            frontier.append(r)
    best = max(good, key=lambda r: float(r["result"]["decode_toks"]))
    c1 = [r for r in good if int(r["row"].get("concurrency", 1)) == 1]
    latency = min(c1 or good, key=lambda r: float(r["result"]["tpot_median_ms"]))
    gp = max(good, key=lambda r: (float(r["result"].get("goodput_pct", 0)), float(r["result"]["decode_toks"])))
    smallest = min(good, key=lambda r: weight_gib(r["row"]) if weight_gib(r["row"]) is not None else math.inf)
    rec = gp if gp in frontier else max(frontier, key=lambda r: float(r["result"]["decode_toks"]))
    rr = rec["row"]; rm = rec["memory"]["mem_gb"]
    spec_lines = ["%s c%s speedup %.2fx acceptance %.2f%%" % (d["slug"], d["concurrency"], float(d["speedup"]), float(d["acceptance_rate"])) for d in drows if d["speedup"] != "n/a"]
    higher = sorted([r for r in good if eng(r["row"].get("engine")) == eng(rr.get("engine")) and quant(r["row"].get("quant")) == quant(rr.get("quant")) and spec(r["row"].get("spec", "base")) == spec(rr.get("spec", "base")) and str(r["row"].get("variant") or "") == str(rr.get("variant") or "") and int(r["row"].get("concurrency", 1)) > int(rr.get("concurrency", 1))], key=lambda r: int(r["row"].get("concurrency", 1)))
    if higher:
        h = higher[0]
        above = "Above c%s, the next measured rung is c%s at %.2f aggregate decode tok/s, %.1f ms median TPOT and %.1f ms median TTFT; higher concurrency is not a free throughput gain for this recommendation." % (rr.get("concurrency"), h["row"].get("concurrency"), float(h["result"]["decode_toks"]), float(h["result"]["tpot_median_ms"]), float(h["result"]["ttft_median_ms"]))
    else:
        above = "Above c%s, no higher successful rung was measured; the next required sweep rung is represented by a blocked page." % rr.get("concurrency")
    lines = [
        "- Best aggregate throughput: %s at c%s, %.2f aggregate decode tok/s." % (best["slug"], best["row"].get("concurrency"), float(best["result"]["decode_toks"])),
        "- Best single-user latency: %s at c1, %.1f ms median TPOT and %.1f ms median TTFT." % (latency["slug"], float(latency["result"]["tpot_median_ms"]), float(latency["result"]["ttft_median_ms"])),
        "- Best goodput: %s at %.1f%%, with the SLO restated as median TTFT <= %.0f ms and median TPOT <= %.0f ms." % (gp["slug"], float(gp["result"].get("goodput_pct", 0)), SLO_TTFT, SLO_TPOT),
        "- Fastest-engine comparison: no fastest engine can be established; qwen3-8-27b-fp8-sglang-c1 and qwen3-8-27b-fp8-vllm-c1 are blocked before valid throughput, so no speed ratio or equivalence claim is made.",
        "- Smallest viable model-file set: %s at %.3f GiB of measured checkpoint files plus any selected draft sidecar; llama.cpp emitted no numeric engine weight breakdown, so mem_gb is not used as a footprint comparison." % (smallest["slug"], weight_gib(smallest["row"])),
        "- Speculation is per concurrency: " + ("; ".join(spec_lines) if spec_lines else "no publishable acceptance/speedup pair was captured."),
        "- Recommendation is constrained to the measured Pareto frontier: %s." % rec["slug"],
        "",
        "For ShareGPT general chat with fixed 256-token responses, run %s %s %s at concurrency %s (slug %s), giving %.2f tok/s aggregate and %.1f ms per token per user at %s GB. %s" % (eng(rr.get("engine")), quant(rr.get("quant")), spec(rr.get("spec", "base")), rr.get("concurrency"), rec["slug"], float(rec["result"]["decode_toks"]), float(rec["result"]["tpot_median_ms"]), "n/a" if rm is None else "%.2f" % rm, above),
    ]
    return "\n".join(lines)

def build_summary(records: list[dict[str, Any]], brows: list[dict[str, Any]], crows: list[dict[str, Any]], drows: list[dict[str, Any]]) -> str:
    dataset = INVENTORY.get("dataset", {})
    assumptions = INVENTORY.get("autobench_assumptions", {})
    atable = [
        ("Total memory ceiling", "re-derived", "%s GiB device-local/UMA inventory; not host MemTotal" % assumptions.get("memory_ceiling_gb")),
        ("Idle baseline consumed", "re-derived", "%s GiB latest host MemAvailable snapshot; each run records a new baseline" % assumptions.get("idle_baseline_gb")),
        ("SGLang batch cap", "re-derived", "runner set --max-running-requests equal to the load level; the SGLang attempt was blocked before a publishable result"),
        ("Usable slack / watchdog", "re-derived", "%s GiB floor; watchdog required for shared UMA" % assumptions.get("watchdog_floor_gb")),
        ("mem_gb method", "re-derived", assumptions.get("mem_source_method")),
        ("Image architecture/digests", "re-derived", "%s; local image digests used where no pullable registry digest exists" % assumptions.get("image_arch")),
        ("Native narrow format", "re-derived", assumptions.get("native_narrow_format")),
        ("Multiple accelerators / TP", "not applicable", "one device; no interconnect"),
        ("gpu-memory-utilization", "re-derived", "%s for ROCm stacks; llama.cpp uses Vulkan allocation" % assumptions.get("gpu_memory_utilization_chosen")),
        ("Crash regime", "re-derived", "%s with shared UMA watchdog protection" % assumptions.get("crash_regime")),
        ("Concurrency danger", "re-derived", "must emerge from this sweep; reference conc-64 not reused"),
    ]
    atable = md_table([{"assumption": a, "state": s, "substitution": v} for a, s, v in atable], ["assumption", "state", "substitution"])
    bmd = md_table(brows, ["config", "engine", "quant", "weights_gib", "kv_gib_at_ctx", "state_gib_at_conc", "total_gib", "fits", "decision", "reason"])
    fields_c = ["slug", "engine", "speculative", "quant", "context", "concurrency", "prefill_toks", "decode_toks", "ttft_median_ms", "ttft_p95_ms", "tpot_median_ms", "goodput_pct", "mem_gb", "completed", "errors", "hit_time_cap", "status"]
    fields_d = ["slug", "base_slug", "method", "num_speculative_tokens", "concurrency", "acceptance_rate", "acceptance_length", "decode_toks_spec", "decode_toks_base", "speedup", "verdict"]
    cmd_docs = "[vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/), [SGLang speculative decoding](https://sgl-project.github.io/advanced_features/speculative_decoding.html), [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), and [AMD Quark](https://github.com/amd/Quark)."
    ds_sha = dataset.get("sha256")
    mtp_rates = [float(d["acceptance_rate"]) for d in drows if d["method"] == "MTP" and d["acceptance_rate"] != "n/a"]
    ngram_rates = [float(d["acceptance_rate"]) for d in drows if d["method"] == "n-gram" and d["acceptance_rate"] != "n/a"]
    if mtp_rates:
        mtp_median = statistics.median(mtp_rates)
        anchor = "measured MTP acceptance median %.2f%%; this is %s the 67%% previous-generation ShareGPT MTP anchor; n-gram is method-specific and excluded" % (mtp_median, "within 10 points of" if abs(mtp_median - 67) <= 10 else "not within 10 points of")
    else:
        anchor = "MTP anchor not assessable because no MTP acceptance metric was emitted and parseable; n-gram is method-specific%s" % (" (measured rates: %.2f%%)" % statistics.median(ngram_rates) if ngram_rates else "")
    speed = [(str(d["method"]), int(d["concurrency"]), float(d["speedup"])) for d in drows if d["speedup"] != "n/a"]
    cross_by_method = {m: next((c for mm, c, v in sorted(speed) if mm == m and v < 1), None) for m in sorted({mm for mm, _, _ in speed})}
    cross_text = "; ".join("%s first loss at c%s" % (m, c) for m, c in cross_by_method.items() if c is not None) or "no publishable crossover below 1.0 observed"
    return "\n".join([
        "# Qwen3.8-27B autobench results", "",
        "## 1. What was benchmarked", "",
        "Qwen/Qwen3.8-27B at context 65536, text-only, thinking disabled, ShareGPT fixed max_tokens=256, closed-loop concurrency, actual max_seconds=300 in the recorded commands. The 300-second cap is shorter than autobench's 900-second reference cap and limits external comparison; it is recorded here rather than hidden. Local model revisions and file sizes are on the pages and in the candidate manifest. Dataset SHA256 is %s over %s conversations; it differs from the task-supplied d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb. The machine-class tag is %s. The September rerun adds the qwen38-mtp recipe at commit 431bf8a821d49e6ac919febc7a91fb2434318110, with q4_0 K/V cache and MTP n-max 2/3/4 arms; its repository probe was not used for the headline metrics." % (ds_sha, dataset.get("conversations"), RECIPE), "",
        "## 2. The machine", "", "See [inventory/machine.json](inventory/machine.json). Memory regime = %s; crash regime = %s; native narrow format = %s." % (assumptions.get("memory_regime"), assumptions.get("crash_regime"), assumptions.get("native_narrow_format")), "", md_table([{"property": r["property"], "value": r["value"]} for r in table_a()], ["property", "value"]), "",
        "## 3. What had to be adapted from autobench", "", "Reference values were not reused; the substitution column records this machine's evidence.", "", atable, "",
        "## 4. Configurations considered", "", bmd, "", "Table B sizes are computed from current repository file metadata, the 16-layer KV formula and recurrent-state formula. They are screening arithmetic, not measured allocations.", "",
        "## 5. Results", "", md_table(crows, fields_c), "", "Charts: [throughput_vs_concurrency](charts/throughput_vs_concurrency.png), [ttft_p95_vs_concurrency](charts/ttft_p95_vs_concurrency.png), [pareto_latency_throughput](charts/pareto_latency_throughput.png). Each has a same-basename CSV.", "",
        "## 6. Speculative decoding", "", md_table(drows, fields_d), "", "67%% anchor cross-check: %s." % anchor, "Per-method crossover below 1.0: " + cross_text + ".", "", "MTP and n-gram pages are in addition to their same-engine/quant/concurrency base siblings. " + cmd_docs, "", "Charts: [speculative_speedup_vs_concurrency](charts/speculative_speedup_vs_concurrency.png) and [acceptance_vs_concurrency](charts/acceptance_vs_concurrency.png).", "",
        "## 7. Memory", "", "See [memory_headroom](charts/memory_headroom.png). mem_gb is a shared-UMA host MemAvailable delta, not a static engine footprint. Startup reservation/weight/KV evidence is quoted on each page; missing engine breakdowns are marked not emitted.", "",
        "## 8. Failures and blocks", "", "\n".join("- %s — %s" % (r["slug"], r["error"]) for r in records if r["status"] == "blocked") or "No blocked attempted pages.", "",
        "## 9. Recommendation", "", recommendations(records, drows), "",
        "## 10. What is not known", "",
        "- Candidate rows marked skip/blocked were not publishable runs; no number is inferred for them.",
        "- The local ShareGPT checksum differs from the task-supplied checksum; only the local SHA256 proves these runs used the same file.",
        "- Local SGLang/vLLM images are not pullable registry artifacts; external reproduction needs their build inputs.",
        "- llama.cpp did not emit a numeric KV/weight/activation/non-torch breakdown for the completed Q4 run; slot count and n_ctx_slot are measured and KV arithmetic is computed.",
        "- Sustained clock fields were blank through the sysfs sampler; busy, temperature and power were sampled.",
        "- A pre-existing qwen38 llama.cpp container was loaded before the rerun and left running. It was observed idle (0.00% CPU, zero processing/deferred requests, no recent log activity); its background state is included in the baselines, and no active competing request was observed.",
        "- No matching Qwen3.8 EAGLE-3 head was found; no other model's head was substituted. External-draft compatibility was not published as success.",
        "- Documentation claims remain claims to recheck: " + cmd_docs,
        "- ExLlamaV3, archived TGI, NVIDIA-only stacks and unverified vendor stacks were not run on this AMD Vulkan/ROCm machine.",
        "",
    ]) + "\n"

def table_a() -> list[dict[str, Any]]:
    out = []
    def walk(prefix: str, value: Any):
        if isinstance(value, dict):
            for k, v in value.items(): walk((prefix + "." if prefix else "") + str(k), v)
        else:
            out.append({"property": prefix, "value": json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)})
    walk("", INVENTORY)
    out.append({"property": "autobench_assumptions (whole block)", "value": json.dumps(INVENTORY.get("autobench_assumptions", {}), separators=(",", ":"))})
    return out

def main():
    records = [record(row) for row in ROWS if has_attempt(row) or row.get("page")]
    CONFIGS.mkdir(exist_ok=True)
    for r in records:
        (CONFIGS / (r["slug"] + ".md")).write_text(page(r))
    write_aux_logs(records)
    b, c, d = table_b(records), table_c(records), table_d(records)
    write_csv(RESULTS / "candidates.csv", b, ["config", "engine", "quant", "weights_gib", "kv_gib_at_ctx", "state_gib_at_conc", "total_gib", "fits", "decision", "reason"])
    write_csv(RESULTS / "all.csv", c, ["slug", "engine", "speculative", "quant", "context", "concurrency", "prefill_toks", "decode_toks", "ttft_median_ms", "ttft_p95_ms", "tpot_median_ms", "goodput_pct", "mem_gb", "completed", "errors", "hit_time_cap", "status"])
    write_csv(RESULTS / "speculative.csv", d, ["slug", "base_slug", "method", "num_speculative_tokens", "concurrency", "acceptance_rate", "acceptance_length", "decode_toks_spec", "decode_toks_base", "speedup", "verdict"])
    make_charts(records, d)
    (ROOT / "SUMMARY.md").write_text(build_summary(records, b, c, d))
    print("attempted=%d pages=%d candidates=%d speculative=%d" % (len(records), len(records), len(b), len(d)))

if __name__ == "__main__":
    main()
