#!/usr/bin/env python3
"""Aggregate repeated autobench-style runs without hiding their raw evidence."""
import argparse
import json
import statistics
import sys


FIELDS = (
    "duration_s",
    "prompt_tokens",
    "completion_tokens",
    "prefill_toks",
    "decode_toks",
    "ttft_median_ms",
    "tpot_median_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "tpot_p95_ms",
    "goodput_pct",
    "req_throughput",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples-output")
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    if len(args.inputs) < 2:
        ap.error("at least two repeat JSON files are required")
    repeats = []
    for path in args.inputs:
        with open(path) as f:
            item = json.load(f)
        item["source_file"] = path
        repeats.append(item)
    decodes = [float(item["decode_toks"]) for item in repeats]
    spread_pct = 0.0 if min(decodes) == 0 else 100.0 * (max(decodes) - min(decodes)) / min(decodes)
    if len(repeats) == 2 and spread_pct > 5.0:
        print(
            "repeat decode throughput spread is %.2f%%; run a third repeat before publishing"
            % spread_pct,
            file=sys.stderr,
        )
        return 3
    out = {
        "slug": args.slug,
        "aggregate_method": "median across independent repeats; raw repeat objects retained",
        "repeat_count": len(repeats),
        "repeat_spread_decode_pct": round(spread_pct, 2),
        "stable_within_5pct": spread_pct <= 5.0,
        "repeats": repeats,
    }
    for field in FIELDS:
        values = [item[field] for item in repeats if item.get(field) is not None]
        if values:
            value = statistics.median(values)
            out[field] = round(value, 3) if isinstance(value, float) else value
        else:
            out[field] = None
    out["completed"] = round(statistics.median(item["completed"] for item in repeats))
    out["errors"] = round(statistics.median(item["errors"] for item in repeats))
    out["concurrency"] = repeats[0]["concurrency"]
    out["hit_time_cap"] = any(item.get("hit_time_cap", False) for item in repeats)
    out["slo_ttft_ms"] = repeats[0].get("slo_ttft_ms", 2000.0)
    out["slo_tpot_ms"] = repeats[0].get("slo_tpot_ms", 50.0)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    if args.samples_output:
        sample_repeats = []
        for repeat_path in args.inputs:
            sample_path = repeat_path.replace(".json", ".samples.json")
            try:
                with open(sample_path) as f:
                    sample_repeats.append({"source_file": sample_path, **json.load(f)})
            except FileNotFoundError:
                pass
        with open(args.samples_output, "w") as f:
            json.dump({"slug": args.slug, "repeats": sample_repeats}, f, indent=2)
            f.write("\n")
    print(
        "aggregate slug=%s repeats=%d spread_decode_pct=%.2f stable=%s"
        % (args.slug, len(repeats), spread_pct, out["stable_within_5pct"])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

