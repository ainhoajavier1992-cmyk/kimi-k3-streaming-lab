# Kimi K3 Public Information

Last checked: 2026-07-22.

This document separates public Kimi K3 information that is already available
from the exact self-hosting metadata still needed for a Colibri-style streaming
runtime.

## Official Public Sources

- Kimi K3 API guide: https://platform.kimi.ai/docs/guide/kimi-k3-quickstart
- Kimi model list: https://platform.kimi.ai/docs/models.md
- Kimi K3 tech blog: https://www.kimi.com/blog/kimi-k3

## Already Public

The official Kimi docs/blog currently state:

- API model id: `kimi-k3`
- Total scale: 2.8 trillion parameters
- Context window: 1 million tokens
- Modalities: native visual understanding in addition to text
- Attention: Kimi Delta Attention (KDA)
- Residual design: Attention Residuals (AttnRes)
- MoE design: Stable LatentMoE
- Expert sparsity: 16 active experts out of 896
- Training/deployment quantization: MXFP4 weights with MXFP8 activations
- Full model weights: promised by July 27, 2026
- More exact architecture/training/evaluation details: promised with the Kimi K3
  technical report

This is enough to confirm that a Colibri-style expert-streaming project is worth
building. It is not enough to implement a faithful local runtime yet.

## Still Not Public Enough For This Runtime

For local inference, the engine needs exact files and tensor metadata:

- `config.json`
- tokenizer files and chat template
- modeling/configuration source, unless upstream Transformers supports K3
- `model.safetensors.index.json` or equivalent weight map
- exact tensor names
- exact layer schedule
- exact router math beyond the high-level "16 of 896" summary
- exact KDA/Gated MLA/KV-cache tensor layout
- exact AttnRes wiring
- exact checkpoint format and scale layout
- license terms for redistribution and conversion

## Hugging Face Check

The official `moonshotai` Hugging Face model listing did not show a Kimi K3 model
repository at this check.

The public K3-related Hugging Face repositories found by search were not usable
model repositories:

- `audnai/penclaw-Kimi-K3.0-abliterated-GGUF`: only `.gitattributes` and
  `README.md` were present at this check.
- `HFVwr/kimi-k3-article-svg-preview`: article and preview assets, not weights
  or model config.

The earlier `reteetzad/Kimi-K3` page behaved like a release placeholder in
previous checks, but the unauthenticated Hugging Face API now returns `401` for
that path. Either way, it does not currently provide usable local-runtime
artifacts.

## Project Implication

We should keep building infrastructure that does not require K3 weights:

- C runtime modularity
- safetensors metadata reader
- converter planning and tensor-map validation
- synthetic oracle tests
- disk I/O benchmarks
- documentation and issue templates

We should wait for official model artifacts before implementing the exact K3
adapter or downloading full weights.
