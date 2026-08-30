# GLM-5.3-Flash on 3x CMP 170HX (SM80) — incompatibility report

**Verdict: BLOCKED — do not download, build, or launch.** The delegation's stop
condition ("if none exists... document that incompatibility and stop instead of
forcing it") is met on two independent grounds.

## Environment verdict (preflight, read-only, 2026-08-30 ~03:14-03:28 UTC)

- All three CMP 170HX cards (GA100, 10de:20c2) visible and healthy inside VM 215
  agent-sandbox: 30-38 C, 37-51 W idle, no Xid in dmesg, driver 610.43.03.
  The earlier "GPU0 Unknown Error / GPU1-GPU2 missing" monitor report is stale.
- GPUs were NOT free: `vllm serve /model` (served name `dsv4s`, DeepSeek-V4-Flash,
  container `dsv4-a100`, PP=3) held 52-64 GiB per card. Preserved untouched.
- `/library` verified: NFSv4 from `192.168.2.18:/library`, 8.4 TB free. HF cache
  contains DeepSeek-V4-Flash-0731 + two Qwen3.8-27B checkpoints; no GLM files.
  Any GLM run would have needed a fresh ~200 GB download (blocked by the stop
  gate; not attempted).

## Blocker 1 — no SM80-capable runtime exists for this architecture

- Architecture `glm5_next` (`Glm5NextForConditionalGeneration`) is **absent
  from upstream vLLM's model registry on main** (checked 2026-08-30, grep
  `glm5` in `vllm/model_executor/models/registry.py`: 0 matches;
  `artifacts/feasibility/vllm-registry-glm5-match-count.txt`).
- The only GLM-5.3-Flash support PR (`vllm-project/vllm#53906`) is **open,
  unmerged, and SM90+ only** (`artifacts/feasibility/vllm-pr-53906.json`).
- The source repo's own DFlash2/EXL3 overlays pin SM121 (`TORCH_CUDA_ARCH_LIST
  =12.1a`, arm64 GB10 images, `sm_121a` cubins). `exl3/overlay/exl3.py:480`
  notes LinearEXL3 needs "CUDA >= Ampere", but that is a capability gate, not a
  working SM80 stack: the repo's sparse-MLA attention path targets SM12x
  FlashInfer backends, absent on SM80. Rebuilding EXL3 kernels for sm_80 and
  substituting a different attention backend is original engineering, out of
  scope, and undocumented upstream.

## Blocker 2 — no checkpoint that fits 3x 64 GiB even if a runtime existed

HF search (`artifacts/feasibility/hf-models-search.json`) yields these GLM-5.3-Flash quants:

| checkpoint | format | size | fits 3x64 GiB? |
|---|---|---|---|
| LibertAIDAI/GLM-5.3-Flash-NVFP4 | NVFP4 | ~181 GB | no (SM121-only runtime anyway) |
| Mia-AiLab/...-EXL3-TR3-4bpw | EXL3/TR3 4bpw | ~164 GiB DL / 176 GB | runtime blocker 1 |
| cyankiwi/GLM-5.3-Flash-AWQ-INT4 | AWQ INT4 (pack-quantized, group 32) | **198.1 GiB** (212.7 GB, 43 shards) | **no — see fit math** |
| unsloth & official FP8/BF16/GGUF variants | FP8/BF16/GGUF | >= 328 GB | no |

AWQ INT4 fit math (TP=3): 212,721,952,636 B / 3 = 66.0 GiB weights per GPU vs
64.0 GiB physical — **-2.04 GiB margin before CUDA context, activations, or any
KV cache** (`artifacts/feasibility/fit-math.txt`). Pipeline parallelism does not
rescue this: splits are layer-wise, not weight-proportional.

## What was verified live, per delegation order

1. Read-only preflight only — no reboot, no reset, no passthrough/power changes,
   no downloads, no builds, no launches. DeepSeek-V4-Flash setup preserved.
2. /library confirmed as the only sanctioned weight/cache location; empty of GLM.
3. GLM repo read in full (root, start.sh, benchmark.py, prefill-benchmark.py,
   dflash2/, exl3/, results/) — methodology extracted in
   `artifacts/feasibility/pi-research-report.md` (Q5/Q6) for future re-use once
   an SM80-viable checkpoint/runtime exists.
4. Qwen3.8-27B PR #1 methodology reviewed; usage-token-counted approach is the
   template this repo should adopt for any future CMP benchmarking.

## Unblocking paths (for maintainers)

1. Land an SM80-capable `glm5_next` backend in upstream vLLM (or merge + extend
   PR #53906 beyond SM90+). Substantial: the NoPE-MLA sparse attention path has
   no SM80 implementation today.
2. Produce a W4A16/AWQ checkpoint that actually fits: e.g. a 3-bit or 2.x-bit
   quant (~130-150 GB), or a true 4-bit with group-0 exclusions yielding
   <= ~55 GiB per GPU at TP=3 after KV/CUDA overhead.
3. Re-run this benchmark plan when either lands; the methodology artifacts here
   are already reusable.

---
Raw evidence: `artifacts/preflight/`, `artifacts/feasibility/`.
Generated 2026-08-30 by Codex (controller) + Pi (repo research), PixelML.
