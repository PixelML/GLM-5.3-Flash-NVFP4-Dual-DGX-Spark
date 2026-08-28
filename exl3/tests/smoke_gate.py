#!/usr/bin/env python3
"""Run exact-response, code-quality, and repetition-loop smoke gates."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


def chat(url: str, key: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    started = time.monotonic()
    response = json.load(urllib.request.urlopen(request, timeout=300))
    message = response["choices"][0]["message"]
    return {
        "content": (message.get("content") or "").strip(),
        "finish_reason": response["choices"][0].get("finish_reason"),
        "usage": response.get("usage", {}),
        "wall_s": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8889/v1/chat/completions")
    parser.add_argument("--model", default="apollo-glm-5.3-flash")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    key = Path(args.api_key_file).read_text().strip()
    exact = chat(args.url, key, args.model, "Reply with exactly GLM_EXL3_OK", 64)
    exact["passed"] = exact["content"] == "GLM_EXL3_OK"

    code = chat(
        args.url,
        key,
        args.model,
        "Write a Python function is_balanced(text: str) -> bool that validates "
        "balanced parentheses. Include at least two assert examples. Output code only.",
        512,
    )
    loop_match = re.search(r"([^\s])\1{24,}", code["content"])
    code["passed"] = (
        "def is_balanced" in code["content"]
        and code["content"].count("assert ") >= 2
        and loop_match is None
    )
    code["repetition_loop"] = loop_match.group(0)[:80] if loop_match else None

    result = {"passed": exact["passed"] and code["passed"], "exact": exact, "code": code}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    print(rendered)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
