#!/usr/bin/env python3
"""Functional gates plus comparable e2e and structural decode throughput."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path


def request_json(url: str, key: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)


def red_png() -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    size = 32
    rows = b"".join(b"\x00" + bytes((255, 0, 0)) * size for _ in range(size))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


def chat(base_url: str, key: str, payload: dict) -> tuple[dict, float]:
    started = time.perf_counter()
    result = request_json(f"{base_url}/chat/completions", key, payload)
    return result, time.perf_counter() - started


def stream_once(base_url: str, key: str, payload: dict) -> dict:
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first = None
    usage = {}
    with urllib.request.urlopen(request, timeout=1200) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if first is None and any(delta.get(name) for name in (
                    "content", "reasoning", "reasoning_content", "tool_calls"
                )):
                    first = time.perf_counter()
    finished = time.perf_counter()
    tokens = int(usage.get("completion_tokens", 0))
    ttft = (first or finished) - started
    decode_seconds = max(finished - (first or finished), 1e-9)
    return {
        "tokens": tokens,
        "seconds": finished - started,
        "ttft": ttft,
        "decode_tps": max(tokens - 1, 0) / decode_seconds,
    }


def benchmark(base_url: str, key: str, model: str, concurrency: int, tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": (
            "Write a compact Python function that validates a topological ordering. "
            "Return code only."
        )}],
        "max_tokens": tokens,
        "temperature": 0,
        "ignore_eos": True,
        "reasoning_effort": "low",
    }
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        runs = list(pool.map(
            lambda _: stream_once(base_url, key, payload), range(concurrency)
        ))
    wall = time.perf_counter() - started
    total = sum(run["tokens"] for run in runs)
    return {
        "concurrency": concurrency,
        "completion_tokens": total,
        "wall_seconds": round(wall, 3),
        "aggregate_e2e_tps": round(total / wall, 2),
        "mean_stream_decode_tps": round(
            sum(run["decode_tps"] for run in runs) / len(runs), 2
        ),
        "mean_ttft_seconds": round(sum(run["ttft"] for run in runs) / len(runs), 3),
    }


def preview(value: object, limit: int = 180) -> str:
    return "" if value is None else str(value).replace("\n", " ")[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="LibertAIDAI/GLM-5.3-Flash-NVFP4")
    parser.add_argument("--secret-file", default=".vllm-api-key")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument(
        "--include-vision",
        action="store_true",
        help="run the opt-in image gate; this needs additional UMA headroom",
    )
    args = parser.parse_args()
    key = Path(args.secret_file).read_text(encoding="utf-8").strip()

    if not args.bench_only:
        models = request_json(f"{args.base_url}/models", key)
        print(json.dumps({"models": [item["id"] for item in models["data"]]}))

        coding, elapsed = chat(args.base_url, key, {
            "model": args.model,
            "messages": [{"role": "user", "content": (
                "Implement binary search in Python. Return code only."
            )}],
            "max_tokens": 256,
            "temperature": 0,
            "reasoning_effort": "low",
        })
        message = coding["choices"][0]["message"]
        used = int(coding.get("usage", {}).get("completion_tokens", 0))
        print(json.dumps({
            "coding": preview(message.get("content")),
            "completion_tokens": used,
            "seconds": round(elapsed, 3),
            "e2e_output_tps": round(used / elapsed, 2),
        }))

        reasoning, _ = chat(args.base_url, key, {
            "model": args.model,
            "messages": [{"role": "user", "content": (
                "A shop discounts $80 by 25%. What is the final price?"
            )}],
            "max_tokens": 512,
            "temperature": 0,
            "reasoning_effort": "high",
        })
        reasoning_message = reasoning["choices"][0]["message"]
        print(json.dumps({
            "reasoning_present": bool(
                reasoning_message.get("reasoning")
                or reasoning_message.get("reasoning_content")
            ),
            "answer": preview(reasoning_message.get("content")),
        }))

        tools, _ = chat(args.base_url, key, {
            "model": args.model,
            "messages": [{"role": "user", "content": "What is the weather in Hanoi?"}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }}],
            "tool_choice": "auto",
            "max_tokens": 256,
            "temperature": 0,
            "reasoning_effort": "low",
        })
        calls = tools["choices"][0]["message"].get("tool_calls", [])
        print(json.dumps({"tool_call_names": [call["function"]["name"] for call in calls]}))

        if args.include_vision:
            vision, _ = chat(args.base_url, key, {
                "model": args.model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What is the dominant color? One word."},
                    {"type": "image_url", "image_url": {"url": red_png()}},
                ]}],
                "max_tokens": 128,
                "temperature": 0,
                "reasoning_effort": "low",
            })
            print(json.dumps({
                "vision": preview(vision["choices"][0]["message"].get("content"))
            }))

    for concurrency in (int(value) for value in args.concurrency.split(",")):
        print(json.dumps({"benchmark": benchmark(
            args.base_url, key, args.model, concurrency, args.output_tokens
        )}))


if __name__ == "__main__":
    main()
