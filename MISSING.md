# What Is Missing

This repository is an early scaffold, not a finished Kimi K3 runtime. This file
exists so contributors can see the real gaps without guessing.

## Not Included On Purpose

- No Kimi K3 weights are included.
- No Colibri source code is vendored or copied.
- No claim is made that this currently runs Kimi K3.

The project is inspired by JustVugg's
[colibri](https://github.com/JustVugg/colibri), but it is an independent
Kimi-oriented implementation scaffold.

## Blocked Until Official Kimi K3 Artifacts Exist

We need the official public release artifacts before the exact architecture can
be locked:

- `config.json`
- `generation_config.json`
- tokenizer files and chat template
- modeling/configuration Python files, unless upstream Transformers supports K3
- `model.safetensors.index.json` or equivalent weight map
- exact tensor names for dense weights, routers, experts, norms, attention, and
  output head
- source checkpoint format details: FP8, BF16, int4, NVFP4, GGUF, or another
  format
- license terms confirming what redistribution and conversion are allowed

Until those exist, we can only build model-independent engine pieces and test
against synthetic fixtures or earlier Kimi-family precedent.

## Missing Runtime Pieces

The current C code proves expert-streaming mechanics with a synthetic MoE model.
These pieces still need real Kimi K3 implementations:

- Kimi K3 config parser and architecture adapter
- real dense weight loader
- real safetensors tensor index reader
- Kimi K3 tokenizer and chat-template path
- real attention/KV-cache implementation
- real router semantics, including any group-top-k or scaling rules
- real sparse-layer schedule
- real expert tensor mapping
- real output head and sampling loop
- long-context memory budgeting
- native speculative decoding support, if K3 exposes an MTP or next-token draft
  head
- OpenAI-compatible server mode

## Missing Converter Work

The current converter is only a planning skeleton. It still needs:

- source snapshot validation
- tensor-name mapping
- dense/routed split
- packed expert writer
- int4/int8/FP8 conversion kernels
- scale format decisions
- resume/checkpoint logic for multi-hundred-GB conversion
- shard integrity checks
- a small converted fixture generated from real K3 tensors

## Missing Correctness Work

Before calling the engine faithful, we need:

- teacher-forcing tests against official Transformers code
- token-exact or tolerance-bounded prefill validation
- route parity tests
- attention parity tests
- quantization error measurements
- small-context and long-context tests
- reproducible decoding tests with fixed seeds

## Missing Performance Work

The current scaffold is intentionally simple. Colibri-class performance still
needs:

- async I/O worker pool
- direct I/O or platform-specific cache-control path
- read coalescing against real K3 expert layout
- learned hot-expert pinning from real route histories
- prefetch/lookahead experiments
- memory-budget planner
- profiling output
- CUDA backend
- Metal backend
- NUMA-aware placement for large CPU hosts

## Missing Project Infrastructure

The repo should still add:

- GitHub Actions CI
- issue templates
- benchmark report template
- contribution checklist for architecture findings
- model-release checklist
- security note for model downloads and untrusted custom code
