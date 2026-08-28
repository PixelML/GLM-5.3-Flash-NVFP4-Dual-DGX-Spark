# DFlash2 evaluation profile (2× DGX Spark)

This profile serves the same `LibertAIDAI/GLM-5.3-Flash-NVFP4` target with
`incoai/GLM-5.3-Flash-DFlash2` as a seven-token block-diffusion draft. It uses
vLLM TP2 over the direct CX7 link and exposes an OpenAI-compatible API on
`:8889`.

The working SM121 port and container come from
[`tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark).
PixelML's contribution here is an independent Apollo reproduction, portable
two-node launch scripts, secret-safe startup, persistent JIT caches, CLIProxy
validation, anti-loop/tool gates, and prompt-shape/prefill measurements.

> [!IMPORTANT]
> The DFlash2 draft checkpoint is licensed CC BY-NC-ND 4.0. Treat this profile
> as non-commercial evaluation only. The base MTP recipe in the repository root
> remains the production-oriented path.

## Pinned stack

| Component | Pin |
|---|---|
| Target | `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `11d73216cd636238e82e1d77fe1042ffab36e7fa` |
| Draft | `incoai/GLM-5.3-Flash-DFlash2` @ `7d74cdd881ed7e32c31175984a67823127b66cfe` |
| Image | `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` |
| Image digest | `sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6` |
| Target weights | ModelOpt NVFP4, Marlin MoE backend |
| KV cache | FP8 E4M3, 3 GiB/rank, 310,292-token pool |
| Context | 262,144 |
| Speculation | DFlash2 block size 8 → `num_speculative_tokens=7` |

## Prerequisites

- Docker with GPU access on both Sparks.
- Passwordless SSH from the head to the worker.
- The target and draft present on local disk on **both** nodes.
- The repository checked out at the same absolute path on both nodes.
- A direct CX7/RoCE link. Override the interface/IP variables if your topology
  differs from Apollo.

The target may be a Hugging Face snapshot symlink. `launch-node.sh` deliberately
mounts the whole HF cache so `config.json` and shard symlinks still resolve
inside Docker.

## Quickstart

On both nodes, expose stable paths for the local checkpoints:

```bash
mkdir -p "$HOME/models"
ln -s /actual/path/to/target-snapshot "$HOME/models/GLM-5.3-Flash-NVFP4"
ln -s /actual/path/to/dflash2-draft "$HOME/models/GLM-5.3-Flash-DFlash2"
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
```

On the head:

```bash
cd dflash2
umask 077
openssl rand -hex 32 > .vllm-api-key

# Apollo defaults shown; override for your fabric.
export WORKER_SSH=apollo-2
export HEAD_IP=10.100.120.2
export WORKER_IP=10.100.120.1
export FABRIC_IF=enp1s0f1np1
export FABRIC_HCA=rocep1s0f1

./start-cluster.sh start
./start-cluster.sh status
```

`VLLM_API_KEY` is read from the mounted secret inside the container, so the
credential does not appear in process arguments or vLLM's startup-argument log.
FlashInfer/vLLM caches persist under
`$HOME/models/vllm-cache-glm53-dflash2`.

## Validate and benchmark

```bash
python3 bench_stream.py --levels 1,2,4 --rounds 2 --max-tokens 400
python3 bench_prompt_shapes.py

python3 ../prefill-benchmark.py \
  --base-url http://127.0.0.1:8889/v1 \
  --model apollo-glm-5.3-flash \
  --secret-file .vllm-api-key
```

The served names are `apollo-glm-5.3-flash` and `glm-5.3-flash`. Full Apollo
measurements and methodology are in
[`../results/APOLLO-2026-08-28-DFLASH2.md`](../results/APOLLO-2026-08-28-DFLASH2.md).

## Operational notes

- Start the worker first; the supplied cluster script waits 20 seconds before
  starting the head.
- Both containers use `--restart no`. A failed boot stays failed and visible.
- `--skip-mm-profiling` is enabled because this is a coding/text lane and GB10
  UMA headroom is more valuable than a maximum-size video dummy forward.
- `--enforce-eager` follows the independently validated SM121 recipe. CUDA
  graphs are not claimed in these numbers.
- DFlash throughput is acceptance-bound. Structured output can exceed 60 tok/s;
  planning-heavy free-form coding was about 25 tok/s in this reproduction.
- Stop both ranks together with `./start-cluster.sh stop`.
