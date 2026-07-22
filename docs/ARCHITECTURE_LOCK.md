# Architecture Lock

This project is the Kimi-side engine scaffold for a Colibri-style MoE streamer.
It starts from the parts that are independent of the final Kimi K3 checkpoint:
expert placement, expert streaming, cache policy, quantized expert matmul, and
release-time validation gates.

Colibrì credit: the reference idea comes from JustVugg's
https://github.com/JustVugg/colibri project. This repository is an independent
OpenClaw-built scaffold for Kimi K3, not an official Colibrì port.

## What Is Locked Now

- Runtime language: C, no Python required for inference.
- Disk expert format: one append-only expert store, fixed-size packed experts,
  one contiguous read per expert.
- Expert payload: three int4 matrices per routed expert: `gate_proj`,
  `up_proj`, `down_proj`, each with per-row float scales.
- Runtime cache: per-layer LRU slots; placement changes speed only.
- Learning cache hook: routed expert usage can be saved and used to preload hot
  experts on a later run.
- Batch-union hook: a layer routes every row in a batch, deduplicates experts,
  and loads each unique expert once before applying it to all rows that chose it.
  The current implementation takes that path only when the layer cache can hold
  the whole union; otherwise it falls back to route-order loads so an undersized
  cache cannot evict an expert before it is applied.
- Router fixture: sigmoid top-k with optional top-k normalization and routed
  scaling, matching the Kimi K2/DeepSeek-V3 family behavior at a systems level.
- Test mode: synthetic MoE model can run end-to-end without Kimi K3 weights.

## What Is Not Locked Until K3 Releases

- Exact `model_type`, `architectures`, and custom modeling file names.
- Exact tensor names and whether language weights are nested under
  `language_model` or directly under `model`.
- Exact MoE layer frequency, number of experts, and number of sparse layers.
- Whether K3 remains text-only DeepSeek-V3-like or adds multimodal wrappers like
  Kimi K2.5/K2.6.
- Attention details, including MLA/Yarn changes and any new sparse attention.
- Native MTP/next-token prediction layers, if any.
- Quantization source format: FP8, compressed-tensors int4, NVFP4, GGUF, or
  something new.

## Current Public Evidence

The Kimi K3 Hugging Face stub at `reteetzad/Kimi-K3` currently has only a model
card and `.gitattributes`; no `config.json`, no safetensors, and no modeling
code. That means we can build and test engine infrastructure now, but we cannot
claim token correctness for K3 yet.

Kimi K2/K2.x public configs are strong precedent:

- `moonshotai/Kimi-K2-Instruct` uses `model_type=kimi_k2`.
- It maps to `DeepseekV3ForCausalLM`.
- It has `hidden_size=7168`, `num_hidden_layers=61`, `first_k_dense_replace=1`.
- It has `n_routed_experts=384`, `num_experts_per_tok=8`,
  `n_shared_experts=1`, `moe_intermediate_size=2048`.
- It uses `scoring_func=sigmoid`, `topk_method=noaux_tc`, `n_group=1`,
  `topk_group=1`, `norm_topk_prob=true`, `routed_scaling_factor=2.827`.
- It has MLA fields: `q_lora_rank`, `kv_lora_rank`,
  `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`.

That is close enough to Colibri/GLM-5.2 to justify building the shared MoE
streaming engine now.

## Release-Time Gates

When K3 is published, do these before downloading full weights:

1. Fetch `config.json`, `generation_config.json`, tokenizer files, and modeling
   code only.
2. Run `tools/kimi_probe.py` to verify the K3 repo has config and weights.
3. Run `tools/kimi_convert_skeleton.py --snapshot <local snapshot>` after a
   metadata-only/local snapshot exists.
4. Confirm top-k router semantics against the official modeling code.
5. Confirm tensor names for routed experts and shared experts.
6. Confirm quantization source format and conversion path.
7. Build a tiny oracle with one or two layers and compare C route/expert output
   against Transformers before any full download.

Only after those gates pass should the full Kimi K3 model be downloaded.
