#!/usr/bin/env python3
"""One real ShareGPT prompt for warm-up and human coherence inspection."""
import argparse
import json
import sys
import urllib.request


def first_human(path):
    with open(path) as f:
        data = json.load(f)
    for conv in data:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "human" and turn.get("value", "").strip():
                return turn["value"]
    raise RuntimeError("dataset has no human prompt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}')
    args = ap.parse_args()
    prompt = first_human(args.dataset)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": json.loads(args.chat_template_kwargs),
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    text = []
    reasoning = []
    usage = None
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("error"):
                    raise RuntimeError("server error chunk: " + str(chunk["error"]))
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
        result = {
            "ok": True,
            "prompt": prompt,
            "response_text": "".join(text),
            "reasoning_text": "".join(reasoning),
            "usage": usage,
        }
    except Exception as exc:
        result = {
            "ok": False,
            "prompt": prompt,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

