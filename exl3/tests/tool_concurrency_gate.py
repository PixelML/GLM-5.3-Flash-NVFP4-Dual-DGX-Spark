#!/usr/bin/env python3
"""Stress required tool arguments under concurrent cold-prefill load."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_action",
            "description": "Open a URL with a browser command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["command", "url"],
            },
        },
    },
]
REQUIRED = {
    "terminal_execute": {"command"},
    "browser_action": {"command", "url"},
}


def run_one(
    url: str,
    model: str,
    api_key: str,
    request_id: int,
    filler_words: int,
    timeout: int,
) -> dict:
    filler = (f" context-{request_id}" * filler_words) if filler_words else ""
    prompt = (
        f"Request {request_id}.{filler}\n"
        "Call terminal_execute exactly once with command='pwd', then call "
        "browser_action exactly once with command='open' and "
        "url='https://example.com'. Make both calls in this turn."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "tool_choice": "required",
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except Exception as error:  # noqa: BLE001
        return {
            "request_id": request_id,
            "ok": False,
            "error": repr(error),
            "wall_s": time.perf_counter() - started,
        }

    calls = result.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
    failures = []
    parsed = []
    for call in calls:
        function = call.get("function") or {}
        name = function.get("name") or ""
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        missing = sorted(REQUIRED.get(name, set()) - set(arguments))
        if not name or missing:
            failures.append({"name": name, "missing": missing, "arguments": arguments})
        parsed.append({"name": name, "arguments": arguments})
    seen = {item["name"] for item in parsed}
    missing_calls = sorted(set(REQUIRED) - seen)
    return {
        "request_id": request_id,
        "ok": not failures and not missing_calls,
        "tool_calls": parsed,
        "argument_failures": failures,
        "missing_calls": missing_calls,
        "prompt_tokens": result.get("usage", {}).get("prompt_tokens"),
        "wall_s": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8889")
    parser.add_argument("--model", default="apollo-glm-5.3-flash")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--filler-words", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out")
    args = parser.parse_args()
    api_key = Path(args.api_key_file).read_text().strip()
    records = []
    for round_index in range(args.rounds):
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as executor:
            futures = [
                executor.submit(
                    run_one,
                    args.url,
                    args.model,
                    api_key,
                    round_index * args.concurrency + index,
                    args.filler_words if index == 0 else 0,
                    args.timeout,
                )
                for index in range(args.concurrency)
            ]
            records.extend(future.result() for future in futures)
    receipt = {
        "concurrency": args.concurrency,
        "rounds": args.rounds,
        "filler_words_on_first_request": args.filler_words,
        "passed": sum(record["ok"] for record in records),
        "failed": sum(not record["ok"] for record in records),
        "records": records,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
