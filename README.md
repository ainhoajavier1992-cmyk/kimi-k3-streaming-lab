# Kimi K3 Streaming Lab

This is a C scaffold for a Colibri-style MoE expert-streaming engine targeting
Kimi K3 once its exact local-runtime metadata and weights are published.

## Credit And Inspiration

This project is inspired by
[colibrì](https://github.com/JustVugg/colibri), the GLM-5.2 expert-streaming
runtime created by **JustVugg**. Colibrì proved the core idea: a very large MoE
model does not need every expert resident in RAM/VRAM at once if the runtime can
place dense weights, cache hot experts, and stream cold experts from disk
faithfully.

This repository is not a fork of Colibrì and does not vendor Colibrì code. It is
an independent Kimi-oriented scaffold built from the same systems idea, with
Colibrì used as the architectural reference.

## How This Was Built

This scaffold was built in an OpenClaw-assisted engineering session. OpenClaw was
used to research Colibrì, inspect Kimi K2/K2.6 public configs as precedent,
write the initial C runtime scaffold, add tests, and document the release gates
for Kimi K3.

The project is intentionally open-ended. The current code proves the
model-independent engine plumbing with synthetic MoE fixtures; contributors are
welcome to replace scaffolding with real Kimi K3 components as soon as official
artifacts exist.

It does not download Kimi K3. Today it builds and runs a synthetic MoE model so
the engine plumbing can be developed safely:

- packed int4 routed experts on disk
- one contiguous expert read per miss
- per-layer LRU cache
- usage stats and hot-expert preload
- batch-union routing so one expert load can serve multiple rows
- sigmoid top-k routing fixture
- SwiGLU expert matmul path
- release-time probes for Kimi metadata

## Build

```bash
make
```

## Smoke Test

```bash
make smoke
```

This generates `fixtures/tiny`, inspects it, and runs streamed inference through
the synthetic MoE path.

Usage stats can seed a warm cache:

```bash
build/k3stream run --model fixtures/tiny --tokens 32 --cache 8 --batch 4 --stats-out usage.txt
build/k3stream run --model fixtures/tiny --tokens 32 --cache 8 --batch 4 --pin usage.txt --pin-per-layer 4
```

## Probe Kimi Metadata

```bash
make probe
```

The probe checks the current public Kimi K3 page without downloading weights and
prints Kimi K2/K2.x architecture precedent.

## Current State

The engine is ready for model-independent work. Kimi has published high-level K3
architecture information: 2.8T parameters, Kimi Delta Attention, Attention
Residuals, Stable LatentMoE, 16 active experts out of 896, native vision, and a
1M-token context window. K3-specific correctness is still blocked until the
local-runtime artifacts exist: a real `config.json`, tokenizer/modeling code,
and a weight index.

See:

- [PROJECT_STATUS.md](PROJECT_STATUS.md)
- [MISSING.md](MISSING.md)
- [ROADMAP.md](ROADMAP.md)
- [docs/KIMI_K3_PUBLIC_INFO.md](docs/KIMI_K3_PUBLIC_INFO.md)
- [docs/ARCHITECTURE_LOCK.md](docs/ARCHITECTURE_LOCK.md)
- [docs/MODEL_DOWNLOAD_HANDOFF.md](docs/MODEL_DOWNLOAD_HANDOFF.md)
- [docs/KIMI_PRECEDENT_SIZE_MATH.md](docs/KIMI_PRECEDENT_SIZE_MATH.md)

## What We Are Waiting For

Do not download full Kimi K3 weights yet. Public API/blog metadata exists, but
the exact model files needed by this runtime are still not available. We are
waiting for:

- official `config.json`
- official modeling/configuration Python files, or upstream Transformers support
- tokenizer files
- `model.safetensors.index.json` or equivalent weight map
- license confirmation

Once those files exist, run:

```bash
make probe
python3 tools/kimi_convert_skeleton.py --snapshot /path/to/kimi-k3-snapshot
```

## Contributions Wanted

This is an open project. Good first areas:

- close items listed in [MISSING.md](MISSING.md)
- follow the staged plan in [ROADMAP.md](ROADMAP.md)
- real Kimi K3 tensor-name mapping once the model is released
- safetensors/FP8/int4 conversion
- exact DeepSeek-V3/Kimi attention path
- tokenizer/chat template support
- async I/O and prefetch
- CUDA/Metal backends
- oracle tests against Transformers
