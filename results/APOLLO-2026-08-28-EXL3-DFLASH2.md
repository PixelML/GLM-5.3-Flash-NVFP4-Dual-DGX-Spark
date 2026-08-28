# Apollo GLM-5.3-Flash EXL3 + DFlash2 validation — 2026-08-28

## Outcome

PixelML reproduced MiaAI's EXL3/TR3 + DFlash2 stack on Apollo and kept it on
the existing authenticated `apollo-glm-5.3-flash` route. The source-shaped
single-stream structured test reached **66.30 decode tok/s**; the comparable
four-stream test reached **154.86 aggregate tok/s**. Realistic speed remained
prompt-dependent: **49.60 tok/s** for code-only and **27.50 tok/s** for
planning-heavy coding at one stream.

The 300K context stress, concurrent required-argument tool calls, image, video,
CLIProxy, and OpenCodex route gates all passed. The deployment is an evaluation
profile because the DFlash2 draft is non-commercial.

## Tested configuration

- 2× NVIDIA DGX Spark, GB10 / SM121, one GPU per node.
- TP2 over Apollo's direct CX7 link (`10.100.120.2 ↔ 10.100.120.1`).
- MiaAI recipe commit: `bd7f55edff9e37b41e1d32e2cf37054fe66d1e58`.
- EXL3 mirror revision: `25a44fdbf16862a46b7cc9921142c6c81350af2f`.
- Original EXL3 snapshot: `brandonmusic/GLM-5.3-Flash-tr3-4bpw@5ab363a8dcf6405955fd5f99671e01a1c9fb124b`.
- DFlash2 revision: `7d74cdd881ed7e32c31175984a67823127b66cfe`.
- Image digest: `sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58`.
- EXL3/TR3 4-bpw routed experts; fused `exl3_moe`; FP8 sparse-MLA KV.
- DFlash2 K=7, CUDA graphs, 900,000 max context, four sequences, vision enabled,
  and Apollo-specific `gpu-memory-utilization=0.88` (upstream 0.87 left only
  13.21 GiB KV versus 13.9 GiB required and estimated 806,400 max tokens).
- Model ID: `apollo-glm-5.3-flash`; authenticated API on head port 8889.

## Cold start and memory

| Item | Apollo observation |
|---|---:|
| Successful launch → API healthy | ~8m40s |
| Post-health shape warmup | 55s |
| Model load | 320.35s |
| Engine warmup | 73.75s |
| KV pool | 942,857 tokens (1.05× the 900K configured request ceiling) |
| Model memory per rank | 82.01 GiB |
| Container restarts / OOM | 0 / 0 on both ranks after the full suite |

The first `gpu-memory-utilization=0.87` launch loaded the model but failed KV
sizing cleanly: 13.21 GiB was available versus 13.9 GiB required. Apollo uses
0.88. The initial 163.65 GiB model download took 18m27s and is not included in
the cold-start number.

## Decode

Temperature 0, thinking off. Decode tok/s excludes TTFT.

| Workload | Concurrency | Aggregate tok/s | Mean stream tok/s | TTFT | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| Structured count 1→200 | 1 | 58.79 | 63.39 | 0.503s | 96.4% |
| Structured count 1→200 | 4 | **154.86** | 41.66 | 0.667s | 95.8% |
| Code only | 1 | 45.71 | **49.60** | 0.539s | 76.9% |
| Code only | 4 | 88.96 | 26.84 | 0.643s | 70.0% |
| Hash-map prose | 1 | 28.30 | 28.30 | 0.540s | 35.5% |
| Planning-heavy coding | 1 | 26.24 | **27.50** | 0.449s | 34.9% |
| Planning-heavy coding | 4 | 47.03 | 16.67 | 4.102s | 38.3% |

Rows use three rounds with unique prefixes and 400 output tokens. A separate
five-run reproduction of MiaAI's source-shaped structured protocol measured a
**66.30 tok/s median** (64.54–68.01), 0.472s TTFT, and 95.6% acceptance. Five
prose runs measured 28.30 tok/s (26.11–30.38). DFlash2 is fast when the draft
can predict the target; prose and planning expose the miss-cost.

## Uncached prefill and long context

| Target | Prompt tokens | TTFT | Client-observed input tok/s | Result |
|---:|---:|---:|---:|---|
| 1K | 1,011 | 1.440s | 702.24 | Pass |
| 4K | 4,151 | 5.372s | 768.21 | Pass |
| 16K | 16,495 | 20.788s | 795.15 | Pass |
| 300K stress | 299,527 | 367.105s | **815.92** | Pass; 0 restarts/OOM |

The short-context rows are medians of three uncached, unique-prefix requests.
The 300K result validates the tested length, not the full 900K ceiling.

## Functional and gateway gates

- Exact response: pass (`GLM_EXL3_OK`).
- Code and 25-character repetition-loop gate: pass.
- Required-argument tool call, single stream: 3/3 pass.
- Required-argument multi-tool concurrency stress: 12/12 pass over three
  four-way rounds. Each round included one 60,248-token cold-prefill request;
  all `terminal_execute` and `browser_action` arguments were intact.
- Image semantic gate: pass (`Red`).
- Video transport/semantic gate: pass on a real AVI URL (9,526 prompt tokens).
- CLIProxy discovery and high-reasoning chat: pass. The verifier's old
  64-token cap caused a false failure because high reasoning consumed the
  budget; 256 tokens returned the exact sentinel and is now the health-check
  setting.
- OpenCodex route: pass; the process is live on port 10100, its `cliproxy`
  provider defaults to `apollo-glm-5.3-flash`, and that local proxy route passed
  the authenticated chat gate.

Machine-readable receipts are in
[`results/raw/APOLLO-2026-08-28-EXL3/`](raw/APOLLO-2026-08-28-EXL3/).

## Comparison with Apollo NVFP4 + DFlash2

| Comparable result | NVFP4 + DFlash2 | EXL3 + DFlash2 | Change |
|---|---:|---:|---:|
| Structured C1 decode | 61.34 tok/s | 66.30 tok/s | +8.1% |
| Code C1 stream decode | 42.55 tok/s | 49.60 tok/s | +16.6% |
| Planning C1 stream decode | 25.28 tok/s | 27.50 tok/s | +8.8% |
| Planning C4 aggregate | 38.54 tok/s | 47.03 tok/s | +22.0% |
| 1K prefill | 1,313.42 tok/s | 702.24 tok/s | −46.5% |
| 4K prefill | 1,403.84 tok/s | 768.21 tok/s | −45.3% |
| 16K prefill | 1,315.46 tok/s | 795.15 tok/s | −39.6% |

EXL3 improved decode and fits a much larger KV pool, but this Apollo build gave
up substantial short-context prefill throughput versus the previous NVFP4
profile. Do not compare structured 1→200 with planning-heavy coding: DFlash2
acceptance, not active-parameter count alone, dominates decode speed.

## Known limits

- The DFlash2 draft is CC BY-NC-ND 4.0: this is non-commercial evaluation.
- The EXL3/TR3 checkpoint is source-available under ShapleyMCG License 1.0,
  not OSI open source. It requires prominent attribution and has a named
  exclusion; review the checkpoint license before deployment.
- [Upstream issue #10](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/10)
  reports occasional blank required tool arguments under
  heavy concurrent cold-prefill load. Apollo's 12/12 gate passed, but this is
  not proof the intermittent issue is fixed. The engine emitted nonfatal
  stop-token grammar-matcher warnings during the stress.
- [Upstream issue #7](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/7)
  reports prefix-cache degradation in long-running agent
  sessions; this short validation did not reproduce or close it.
- A 900K configured ceiling is not evidence that a 900K request was tested.
  This receipt claims only the longest completed stress request.

## Attribution

MiaAI created the dual-Spark EXL3/vLLM overlay and original measurements:
[`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).
Brandon M. Music created the ShapleyMCG EXL3/TR3 checkpoint; Z.AI created the
base model; IncoAI created the DFlash2 draft. PixelML's contribution is this
independent Apollo reproduction, security/topology adaptation, and benchmark
receipt.

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title  = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for
            Low-Bit Mixture-of-Experts Models},
  year   = {2026},
  url    = {https://github.com/brandonmmusic-max/shapleymcg},
  note   = {Licensed under the ShapleyMcg License v1.0}
}
```
