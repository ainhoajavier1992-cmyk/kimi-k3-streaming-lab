# Project Status

## Built So Far

- C runtime scaffold for a Colibri-style MoE expert streamer.
- Synthetic int4 MoE fixture generator.
- Packed expert file format with one contiguous disk read per expert.
- Per-layer LRU cache.
- Usage stats and hot-expert preload.
- Batch-union route deduplication with correctness fallback.
- Kimi public metadata probe.
- Kimi snapshot/converter planning skeleton.
- Regression tests proving cache size and batch-union do not change outputs.

## Attribution

The project is inspired by JustVugg's
[colibrì](https://github.com/JustVugg/colibri), the C runtime that demonstrated
expert streaming for GLM-5.2. This repository is an independent Kimi-oriented
scaffold built from that architectural idea.

## Build Method

The initial scaffold was built with OpenClaw assistance: research, code
generation, test writing, and documentation were done in an OpenClaw/Codex
engineering session. The project is public/open so other developers can repair,
replace, and extend the scaffolding as Kimi K3 artifacts become available.

## Verified

```bash
make test
make smoke
make probe
```

## Waiting For

Kimi K3 is not downloadable yet in a useful form. The public page currently has
no weights, no `config.json`, no tokenizer files, and no safetensors index.

Before full model download, we need:

- `config.json`
- `generation_config.json`
- tokenizer files
- modeling/configuration code, unless upstream Transformers supports it
- `model.safetensors.index.json`
- confirmed license terms

Only then can the real Kimi adapter be locked and validated against a
Transformers oracle.

## Next Engineering Steps After K3 Release

1. Fetch metadata only.
2. Run `make probe`.
3. Run `python3 tools/kimi_convert_skeleton.py --snapshot <snapshot>`.
4. Confirm router, attention, tensor names, quantization source format.
5. Build tiny oracle tests against official modeling code.
6. Implement real Kimi attention and tokenizer path.
7. Convert a small subset, then the full model.
