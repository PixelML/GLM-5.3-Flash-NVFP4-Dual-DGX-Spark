<h1 align="center">GLM-5.3-Flash-NVFP4-Dual-DGX-Spark</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve the **[LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)** checkpoint on a **2× DGX Spark** kit. The repository includes the production-oriented multimodal MTP/Ray profile on **:8888** and a separately licensed [DFlash2 evaluation profile](dflash2/README.md) on **:8889**.

## Hardware / topology

Two GB10 Sparks (SM121, 128 GiB UMA each), one GPU per node, **TP=2** via Ray.

```
spark1 (head)                         spark2 (worker)
10.0.0.1                              zurih@10.0.0.2
Ray head + vLLM API :8888             Ray worker
HF cache (local disk)                 HF cache (rsync’d once)
        CX7 QSFP  (NCCL / RoCE)
  enp1s0f1np1 / rocep1s0f1  ↔  enp1s0f0np0 / rocep1s0f0
```

Both containers run `--network host --ipc=host`. Ray may use the `10.0.0.1`/`10.0.0.2` aliases; **NCCL cannot** — `start.sh` pins the CX7 NICs and IB HCAs so `ncclCommInitRank` does not hang.

**Needs:** Docker (no sudo) on both nodes, passwordless SSH from head → `zurih@10.0.0.2`, `hf`/`huggingface-cli` + `curl` + `rsync` + `python3` on the head, ~200 GiB free per node (~181 GiB weights).

## What you get

| Capability | Default on this kit |
|---|---|
| OpenAI-compatible API | `http://127.0.0.1:8888/v1` (LAN: head IP) |
| Tool calling | `--tool-call-parser glm47 --enable-auto-tool-choice` |
| Reasoning | `--reasoning-parser glm45` |
| Speculative decode | MTP, **4** tokens |
| Multimodal | image + video (`LIMIT_MM={"image":4,"video":1}`) |
| Context | `--max-model-len 262144` (checkpoint is 1M-native) |
| Scheduler | `--max-num-seqs 8`, `--block-size 2304` |
| KV | `--kv-cache-dtype fp8_e4m3` |
| MoE | **marlin** + `--enforce-eager` (GB10-known-good) |

Model name on the wire: `LibertAIDAI/GLM-5.3-Flash-NVFP4`.

## Structural decode tok/s

### DFlash2 evaluation profile

PixelML independently reproduced the public vLLM DFlash2 SM121 port on Apollo.
At single-stream temperature 0 it measured **61.34 decode tok/s** for structured
counting (91.0% draft acceptance), **42.55 tok/s** for code-only output (61.1%
acceptance), and **25.28 tok/s** for planning-heavy coding (30.7% acceptance).
The same endpoint delivered 1.31K–1.40K uncached input tok/s from 1K–16K.

Use [`dflash2/`](dflash2/README.md) for the pinned image and two-node launcher.
Full methodology, cold start, gateway gates, and caveats are in
[`results/APOLLO-2026-08-28-DFLASH2.md`](results/APOLLO-2026-08-28-DFLASH2.md).

> The DFlash2 draft is CC BY-NC-ND 4.0. This profile is non-commercial
> evaluation only; it is not a drop-in replacement for the base MTP profile.

On this recipe (SM90 NoPE + FA2, marlin, MTP-4, fp8_e4m3 KV, 262k, eager). Concurrent streams = `--max-num-seqs` class.

| Concurrency | agg tok/s | tok/s/stream | TTFT |
|---|---|---|---|
| ×1 | 23–30 | 23–30 | 6.54s |
| ×2 | 31–37 | 16–19 | 6.40s |
| ×4 | 43 | 13 | 8.35s |
| ×6 | 64 | 11 | 1.03s † |
| ×8 | 72 | 10 | 6.17s |

† warm

### PixelML Apollo validation

PixelML independently reproduced this recipe on a separate two-Spark cluster
at model revision `11d73216cd636238e82e1d77fe1042ffab36e7fa` and upstream
recipe commit `aed98a1`. Three warm, low-reasoning, fixed-256-token runs per
row measured the following client-observed aggregate output throughput:

| Concurrency | Median agg tok/s | Warm range | Median TTFT |
|---|---:|---:|---:|
| ×1 | 26.55 | 24.10–27.00 | 0.432s |
| ×2 | 43.19 | 42.64–44.50 | 0.538s |
| ×4 | 58.36 | 51.74–58.47 | 0.481s |
| ×6 | 80.57 | 79.58–93.83 | 0.560s |
| ×7 | **82.12** | 72.77–84.84 | 0.499s |
| ×8 | 69.85 | 69.50–73.63 | 2.908s |

Seven streams are the effective no-queue sweet spot for the validated 8.5-GiB
KV layout. The eighth request queues, increasing TTFT and reducing aggregate
throughput. Full methodology, functional gates, cold start, and failure notes
are in [`results/APOLLO-2026-08-27.md`](results/APOLLO-2026-08-27.md).

### Fresh Apollo revalidation

A separate clean pass measured uncached prefill, repeated structural decode,
and a real OpenCodex repository-agent trace through CLIProxy:

| Workload | Result |
|---|---:|
| 1K-token uncached prefill | 1,276.65 input tok/s |
| 4K-token uncached prefill | 1,371.87 input tok/s |
| 16K-token uncached prefill | 1,362.75 input tok/s |
| Single-stream fixed-256 decode | 27.28 output tok/s |
| Best fresh aggregate, 6 streams | 67.55 output tok/s |

All 103 prefill/decode requests finished by length with zero server-reported
prefix-cache hits. The OpenCodex task consumed 111,106 input tokens, made a
coherent multi-step read-only inspection, and returned a 4,002-token repository
assessment without the punctuation loop observed with Qwen3.8-Flash-Next.
The full raw methodology, ranges, route gates, and the newly isolated vision
UMA failure are in
[`results/APOLLO-2026-08-27-REVALIDATION.md`](results/APOLLO-2026-08-27-REVALIDATION.md).

## Quickstart

```bash
umask 077
openssl rand -hex 32 > .vllm-api-key
VLLM_API_KEY="$(<.vllm-api-key)" ./start.sh
```

First run: build/ship the local image if missing → download ~181 GiB into the HF cache → refresh `chat_template.jinja` from Hugging Face (`emit_image` / `emit_video`) → rsync weights to the worker → Ray worker on spark2, Ray head + `vllm serve` on spark1 → poll `/health` up to **3600s** (320B MoE init is slow).

Weights already on disk:

```bash
SKIP_DOWNLOAD=1 ./start.sh
```

```bash
./start.sh status          # containers, API health, Ray cluster
./start.sh logs            # follow head (driver + API)
./start.sh logs worker     # follow worker container
./start.sh stop            # tear down both ranks
./start.sh restart         # stop + start
```

Ctrl-C during the wait detaches; the cluster keeps running. Default is not to tail after ready (`TAIL=1` to keep following).

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H "Authorization: Bearer $(<.vllm-api-key)" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "LibertAIDAI/GLM-5.3-Flash-NVFP4",
    "messages": [{"role": "user", "content": "hello!"}],
    "reasoning_effort": "low"
  }'
```

Image + video use standard OpenAI multimodal content parts (`image_url` /
`video_url`). This checkpoint's refreshed chat template accepts the top-level
OpenAI-compatible `reasoning_effort` field with `low`, `high`, or `max`.
`low` is the direct-answer/coding profile; `high` and `max` expose reasoning in
vLLM's `message.reasoning` field before the final `message.content`.

Run the functional gates and fixed-output concurrency benchmark after the API
is ready:

```bash
./benchmark.py --secret-file .vllm-api-key --concurrency 1,2,4,8
```

The benchmark reports client-observed aggregate output TPS, mean per-stream
decode TPS excluding TTFT, and mean TTFT. Keep the model, output length,
thinking mode, and concurrency identical when comparing profiles. Vision is
intentionally opt-in: the 0.84 text-throughput profile has too little UMA
headroom for a safe first image compile on the Apollo reproduction. Use
`--include-vision` only after lowering the GPU/KV memory budget and validating
the Ray memory margin.

Measure uncached prefill separately with unique prefixes:

```bash
./prefill-benchmark.py --secret-file .vllm-api-key
```

## Image

Serving tag: **`mia/glm53-flash-spark:mm-ray-v1`** (local default; `start.sh` does not change this).

A private copy is on GHCR for MiaAI-Lab members:

```bash
docker pull ghcr.io/miaai-lab/glm53-flash-spark:mm-ray-v1
```

This is a **local** tag (kernel `glm53-flash-sm121:v8` + Ray + multimodal defaults). **Do not `docker pull` it from Docker Hub** — `start.sh` will not pull `mia/glm53-flash-spark:*`; it builds.

If the tag is missing on the head, `start.sh` runs [`files/build.sh`](files/build.sh):

1. Kernel layer **`glm53-flash-sm121:v8`** from [`files/Dockerfile`](files/Dockerfile) (applies [`files/glm53-flash_SM121.py`](files/glm53-flash_SM121.py) **sm90**, seven steps — stock SM12 would take the packed SM120 MLA path, which is wrong for this NoPE checkpoint).
2. Serving layer from [`files/Dockerfile.mm-ray`](files/Dockerfile.mm-ray): Ray 2.58.0, MM env, API **:8888**.
3. `docker save | ssh docker load` onto the worker.

`PULL=1 ./start.sh` rebuilds the local tag and re-ships. You can also build without launching:

```bash
./files/build.sh
```

The kernel layer selects SM90 sparse-MLA + FA2 for GB10; it does **not** invent FlashInfer FA2 or SM121 MLA. `sm120` in `glm53-flash_SM121.py` is an unused packed-cache mode and is **not** the serve path (not baked).

## Configuration

All optional. Defaults match a working 2× Spark + CX7 + Portainer-on-8000 kit.

| Variable | Default | What it does |
|---|---|---|
| `IMAGE` | `mia/glm53-flash-spark:mm-ray-v1` | Serving image |
| `PORT` | `8888` | OpenAI API on the head (8000 is Portainer) |
| `VLLM_API_KEY` | *(unset)* | Protect OpenAI-compatible `/v1` routes; passed as an environment variable so the secret is not written into generated commands or logs |
| `MOE_BACKEND` | `marlin` | `marlin` (default) · `native` · `auto` (native, then marlin on `cudaErrorNoKernelImageForDevice`) |
| `MTP_TOKENS` | `4` | MTP speculative tokens (`0` disables) |
| `MAX_MODEL_LEN` | `262144` | Context; model is 1M-native (`1048576` if you need it) |
| `GPU_MEM_UTIL` | `0.84` | vLLM GPU memory budget (GB10 UMA; **0.90 fails** the free-mem check). Profiles KV when `KV_CACHE_MEMORY` is unset |
| `ENFORCE_EAGER` | `1` | Skip CUDA-graph capture at init |
| `BLOCK_SIZE` | `2304` | Paged-MQA block size |
| `MAX_NUM_SEQS` | `8` | Scheduler concurrency |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV dtype |
| `KV_CACHE_MEMORY` | *(unset)* | Optional `--kv-cache-memory` pin in bytes. Default is empty so `GPU_MEM_UTIL=0.84` sizes KV; set only if MTP-4 UMA-OOMs |
| `LIMIT_MM` | `{"image":4,"video":1}` | Max image/video items per prompt |
| `SKIP_MM_PROFILING` | `1` | Serve MM without a max-size dummy forward at init (avoids UMA OOM) |
| `HEAD_IP` | `10.0.0.1` | Head as seen by the worker |
| `WORKER_SSH` | `zurih@10.0.0.2` | Worker SSH target |
| `WORKER_IP` | `10.0.0.2` | Worker as seen by the head |
| `TP` | `2` | Tensor parallel (1 GPU × 2 nodes) |
| `RAY_PORT` | `6379` | Ray GCS |
| `READY_TIMEOUT` | `3600` | `/health` wait (= `VLLM_ENGINE_READY_TIMEOUT_S`) |
| `EXTRA_ARGS` | — | Extra flags appended to `vllm serve` |
| `SKIP_DOWNLOAD` | `0` | `1` = skip HF download check |
| `SKIP_SYNC` | `0` | `1` = skip rsync to worker |
| `REFRESH_WEIGHTS` | `0` | `1` = re-run `hf download` |
| `HF_HUB_OFFLINE` | — | Skip HF etag checks at startup |
| `PULL` | `0` | `1` = rebuild local image and re-ship |
| `TAIL` | `0` | `1` = keep following head logs after ready |
| `NCCL_DEBUG` | `WARN` | Passed through to both containers |

CX7 pins (`HEAD_CX7_IF`, `WORKER_CX7_IF`, `HEAD_CX7_IB`, `WORKER_CX7_IB`) default to the QSFP pair on this kit.

## Repo layout

| Path | Role |
|---|---|
| [`start.sh`](start.sh) | Start / stop / restart / status / logs |
| [`prefill-benchmark.py`](prefill-benchmark.py) | Unique-prefix uncached prefill benchmark |
| [`files/build.sh`](files/build.sh) | Build kernel v8 + `mm-ray-v1` if missing |
| [`files/Dockerfile`](files/Dockerfile) | Kernel layer `glm53-flash-sm121:v8` |
| [`files/Dockerfile.mm-ray`](files/Dockerfile.mm-ray) | Serving layer (Ray + MM + :8888) |
| [`files/glm53-flash_SM121.py`](files/glm53-flash_SM121.py) | SM121 kernel patch (default **sm90**, seven steps) |
| [`files/chat_template.jinja`](files/chat_template.jinja) | Baked fallback; runtime still refreshes from HF |

`start.sh` writes `.glm53-head.inner.sh` / `.glm53-worker.inner.sh` on every launch (gitignored). `docs/` and `logs/` are gitignored.

## Notes

- **GB10 is UMA.** Weights ~90 GiB per rank; Ray’s default object store would steal RAM from the GPU budget. This recipe uses a 4 GiB Ray object store, `GPU_MEM_UTIL=0.84` (which profiles KV — `--kv-cache-memory` is unset by default), `--enforce-eager`, and `--skip-mm-profiling`. Set `KV_CACHE_MEMORY=<bytes>` only if MTP-4 UMA-OOMs. Do not run another GPU model on either node at the same time.
- **The max-throughput profile is text-first.** On Apollo, the first 32×32 image request crossed Ray's 95% node-memory threshold by about 6.3 MiB; Ray killed TP0 and vLLM exited. Lower `GPU_MEM_UTIL`/the KV budget before production multimodal use. Do not disable Ray's memory monitor as a first fix.
- **Tear down both ranks before relaunch.** Leftover Ray/NCCL on either Spark will fight the next start. Use `./start.sh stop` or `./start.sh restart` — not a head-only `docker rm`.
- **Thinking off:** pass `"chat_template_kwargs": {"enable_thinking": false}` on the chat-completions body.
- **Local image tag.** `mia/glm53-flash-spark:mm-ray-v1` is built here. Do not `docker pull` that tag from Docker Hub.

## License

This repository's code is provided under the [MIT License](LICENSE).
