# Kimi Precedent Size Math

This is not Kimi K3 truth. It is the sizing precedent from public Kimi K2/K2.6
configs, useful for checking whether a Colibri-style engine is the right shape.

## Kimi K2 / K2.6 Text Config

Public configs show:

- `num_hidden_layers=61`
- `first_k_dense_replace=1`
- `n_routed_experts=384`
- `num_experts_per_tok=8`
- `hidden_size=7168`
- `moe_intermediate_size=2048`
- `n_shared_experts=1`
- `scoring_func=sigmoid`
- `topk_method=noaux_tc`
- `n_group=1`
- `topk_group=1`

So the language model has 60 sparse MoE layers and 384 routed experts per sparse
layer.

## Int4 Expert Container Estimate

Each routed expert is a SwiGLU MLP with three matrices:

- `gate_proj`: `[2048, 7168]`
- `up_proj`: `[2048, 7168]`
- `down_proj`: `[7168, 2048]`

With packed int4 weights and one float scale per row:

- `gate_proj`: `2048 * ceil(7168 / 2) + 2048 * 4 = 7,348,224 bytes`
- `up_proj`: same, `7,348,224 bytes`
- `down_proj`: `7168 * ceil(2048 / 2) + 7168 * 4 = 7,368,704 bytes`
- per routed expert: `22,065,152 bytes`, about `22.1 MB`

Total routed expert store:

```text
60 sparse layers * 384 experts * 22,065,152 bytes = 508.4 GB
```

Cold routed-expert read per token:

```text
60 sparse layers * 8 experts/token * 22,065,152 bytes = 10.59 GB/token
```

That is extremely close to Colibri's GLM-5.2 streaming regime. The conclusion is
that a Kimi K3 runner is worth building if K3 keeps a similar DeepSeek-V3-style
MoE layout, even if the exact layer/expert counts change.

## Compressed KV Precedent

Kimi K2/K2.6 expose MLA fields:

- `kv_lora_rank=512`
- `qk_rope_head_dim=64`

A compressed KV cache stores roughly `(512 + 64) * 4 = 2304 bytes` per token per
layer before allocator overhead. Across 61 layers, that is about `140 KB/token`,
much smaller than storing full key/value heads. This is another reason a
Colibri-style memory hierarchy is plausible for Kimi.
