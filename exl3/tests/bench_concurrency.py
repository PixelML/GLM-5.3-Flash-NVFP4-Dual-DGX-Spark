#!/usr/bin/env python3
"""Comparable C1/C2/C4 decode benchmark for Apollo EXL3 + DFlash2."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import time
import urllib.request
import os
from pathlib import Path


PROMPTS = {
    "structured": [
        "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
    ],
    "code": [
        "Output only Python code. Implement a typed thread-safe LRU cache with get/put and four assertions. No prose."
    ],
    "prose": [
        "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough."
    ],
    "planning": [
        "Implement a thread-safe LRU cache in Python with O(1) get and put. Include tests and explain the locking strategy.",
        "Implement a bounded async worker pool in TypeScript with cancellation, backpressure, and unit tests.",
        "Find and fix the race conditions in a typical producer-consumer queue. Give corrected Python code and explain each fix.",
        "Design a PostgreSQL job queue using SKIP LOCKED. Include schema, claim query, retry policy, and operational caveats.",
    ],
}


def request_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def run_one(
    url: str, api_key: str, model: str, prompt: str, max_tokens: int
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=request_headers(api_key),
    )
    start = time.perf_counter()
    first_token_at = None
    last_token_at = None
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            item = json.loads(line[6:])
            usage = item.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
            choices = item.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                now = time.perf_counter()
                first_token_at = first_token_at or now
                last_token_at = now
    end = time.perf_counter()
    if completion_tokens is None:
        raise RuntimeError("stream ended without completion-token usage")
    ttft = (first_token_at or end) - start
    decode_window = max((last_token_at or end) - (first_token_at or end), 1e-9)
    decode_tps = max(completion_tokens - 1, 0) / decode_window
    return {
        "tokens": completion_tokens,
        "wall_s": end - start,
        "ttft_s": ttft,
        "decode_tps": decode_tps,
        "e2e_tps": completion_tokens / max(end - start, 1e-9),
    }


def metric_totals(url: str, api_key: str) -> tuple[float, float]:
    req = urllib.request.Request(f"{url}/metrics", headers=request_headers(api_key))
    text = urllib.request.urlopen(req, timeout=15).read().decode()
    drafted = accepted = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            drafted += float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            accepted += float(line.rsplit(" ", 1)[-1])
    return drafted, accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8889")
    parser.add_argument(
        "--api-key-file",
        default=os.environ.get(
            "API_KEY_FILE", str(Path(__file__).with_name(".vllm-api-key"))
        ),
    )
    parser.add_argument("--model", default="apollo-glm-5.3-flash")
    parser.add_argument("--workload", choices=sorted(PROMPTS), default="planning")
    parser.add_argument("--levels", default="1,2,4")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--out")
    args = parser.parse_args()
    api_key = Path(args.api_key_file).read_text().strip()

    print("warming engine ...", flush=True)
    run_one(
        args.url,
        api_key,
        args.model,
        "Count from 1 to 80, one integer per line, with no commentary.",
        160,
    )

    results = {}
    for concurrency in [int(x) for x in args.levels.split(",")]:
        drafted0, accepted0 = metric_totals(args.url, api_key)
        samples = []
        aggregate_tokens = 0
        aggregate_wall = 0.0
        failures = 0
        for round_index in range(args.rounds):
            workload_prompts = PROMPTS[args.workload]
            prompts = [
                f"[unique {random.randrange(10**12)}] "
                + workload_prompts[
                    (round_index * concurrency + i) % len(workload_prompts)
                ]
                for i in range(concurrency)
            ]
            wave_start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(concurrency) as executor:
                futures = [
                    executor.submit(
                        run_one,
                        args.url,
                        api_key,
                        args.model,
                        prompt,
                        args.max_tokens,
                    )
                    for prompt in prompts
                ]
                for future in futures:
                    try:
                        sample = future.result()
                        samples.append(sample)
                        aggregate_tokens += sample["tokens"]
                    except Exception as error:  # noqa: BLE001
                        failures += 1
                        print(f"request failed: {error}", flush=True)
            aggregate_wall += time.perf_counter() - wave_start
        drafted1, accepted1 = metric_totals(args.url, api_key)
        acceptance = (
            (accepted1 - accepted0) / (drafted1 - drafted0)
            if drafted1 > drafted0
            else None
        )
        row = {
            "requests": len(samples),
            "failures": failures,
            "aggregate_tps": aggregate_tokens / max(aggregate_wall, 1e-9),
            "mean_stream_decode_tps": statistics.mean(
                sample["decode_tps"] for sample in samples
            ),
            "mean_e2e_tps": statistics.mean(sample["e2e_tps"] for sample in samples),
            "mean_ttft_s": statistics.mean(sample["ttft_s"] for sample in samples),
            "draft_acceptance": acceptance,
        }
        results[str(concurrency)] = row
        print(f"C{concurrency}: {json.dumps(row, sort_keys=True)}", flush=True)
    record = {
        "model": args.model,
        "workload": args.workload,
        "max_tokens": args.max_tokens,
        "rounds": args.rounds,
        "results": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
