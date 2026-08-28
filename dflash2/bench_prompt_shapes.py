#!/usr/bin/env python3
"""Measure DFlash2 speed sensitivity to output shape."""

import json
import os
import statistics
from pathlib import Path

from bench_stream import metric_totals, run_one


URL = "http://127.0.0.1:8889"
KEY = Path(
    os.environ.get("API_KEY_FILE", Path(__file__).with_name(".vllm-api-key"))
).read_text().strip()
CASES = {
    "count_1_200": (
        "Count from 1 to 200, one integer per line. Output only the integers.",
        450,
    ),
    "alphabet_x8": (
        "Write the lowercase English alphabet eight times, one alphabet per line, with no labels or commentary.",
        400,
    ),
    "code_only": (
        "Output only Python code. Implement a typed thread-safe LRU cache with get/put and four assertions. No prose.",
        500,
    ),
}


for name, (prompt, max_tokens) in CASES.items():
    drafted0, accepted0 = metric_totals(URL, KEY)
    samples = [run_one(URL, KEY, prompt, max_tokens) for _ in range(3)]
    drafted1, accepted1 = metric_totals(URL, KEY)
    row = {
        "median_decode_tps": statistics.median(x["decode_tps"] for x in samples),
        "median_e2e_tps": statistics.median(x["e2e_tps"] for x in samples),
        "median_ttft_s": statistics.median(x["ttft_s"] for x in samples),
        "tokens": [x["tokens"] for x in samples],
        "draft_acceptance": (
            (accepted1 - accepted0) / (drafted1 - drafted0)
            if drafted1 > drafted0
            else None
        ),
    }
    print(f"{name}: {json.dumps(row, sort_keys=True)}", flush=True)
