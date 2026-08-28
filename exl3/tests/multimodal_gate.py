#!/usr/bin/env python3
"""Small image semantic gate and optional video transport gate."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image


def data_image() -> str:
    image = Image.new("RGB", (32, 32), (255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def chat(url: str, model: str, key: str, content: list[dict]) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 96,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        result = json.load(response)
    message = result.get("choices", [{}])[0].get("message", {})
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning") or message.get("reasoning_content") or "",
        "usage": result.get("usage") or {},
        "wall_s": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8889")
    parser.add_argument("--model", default="apollo-glm-5.3-flash")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--video-url")
    parser.add_argument("--out")
    args = parser.parse_args()
    key = Path(args.api_key_file).read_text().strip()

    image = chat(
        args.url,
        args.model,
        key,
        [
            {"type": "text", "text": "Name only the dominant color in this image."},
            {"type": "image_url", "image_url": {"url": data_image()}},
        ],
    )
    image["passed"] = image["content"].strip().lower().rstrip(".") == "red"
    receipt: dict[str, object] = {"image": image}

    if args.video_url:
        try:
            video = chat(
                args.url,
                args.model,
                key,
                [
                    {
                        "type": "text",
                        "text": "In one short sentence, describe the main visible action in this video.",
                    },
                    {"type": "video_url", "video_url": {"url": args.video_url}},
                ],
            )
            video["passed"] = bool(video["content"].strip())
        except Exception as error:  # noqa: BLE001
            video = {"passed": False, "error": repr(error)}
        receipt["video"] = video

    if args.out:
        Path(args.out).write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
