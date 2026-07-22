# Kimi K3 Streaming Lab

This is a C scaffold for a Colibri-style MoE expert-streaming engine targeting
Kimi K3 once its real config and weights are published.

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

The engine is ready for model-independent work. K3-specific correctness is
blocked until K3 publishes at least a real `config.json`, modeling code, and a
weight index.

See:

- [PROJECT_STATUS.md](PROJECT_STATUS.md)
- [docs/ARCHITECTURE_LOCK.md](docs/ARCHITECTURE_LOCK.md)
- [docs/MODEL_DOWNLOAD_HANDOFF.md](docs/MODEL_DOWNLOAD_HANDOFF.md)
- [docs/KIMI_PRECEDENT_SIZE_MATH.md](docs/KIMI_PRECEDENT_SIZE_MATH.md)

## What We Are Waiting For

Do not download full Kimi K3 weights yet. The current public K3 page is only a
release stub. We are waiting for:

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
