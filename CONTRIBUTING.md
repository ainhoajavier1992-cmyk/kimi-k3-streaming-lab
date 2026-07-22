# Contributing

This project is intentionally open. It exists so people can collectively turn a
Colibri-style MoE expert-streaming scaffold into a real Kimi K3 runtime once the
official Kimi K3 artifacts are public.

## Ground Rules

- Preserve correctness first: placement and cache policy may change speed, but
  must not silently change router semantics or model math.
- Do not add full Kimi K3 weights to this repository.
- Do not download the full model until metadata gates pass.
- Add tests when changing cache, routing, quantization, or batch-union behavior.
- Credit upstream work clearly. This project is inspired by JustVugg's Colibrì,
  but it should remain an independent Kimi-focused implementation unless a
  deliberate fork decision is made.

## How The Initial Scaffold Was Built

The initial repo was created in an OpenClaw-assisted session:

1. Read and studied Colibrì's public repository as the reference design.
2. Checked public Kimi K2/K2.6 configs to verify DeepSeek-V3-style MoE precedent.
3. Built a pure-C synthetic MoE expert streamer with packed int4 experts.
4. Added per-layer LRU caching, usage stats, hot-expert preload, and batch-union.
5. Added tests that verify cache/batch placement changes I/O without changing
   output checksums.
6. Documented the exact Kimi K3 release artifacts still needed.

## What Needs Work After Kimi K3 Releases

- Confirm official architecture and license.
- Implement real Kimi config parsing.
- Implement exact attention and KV-cache behavior.
- Map all routed/shared expert tensor names.
- Build a real converter.
- Validate against a Transformers oracle.
- Optimize I/O and hardware backends.
