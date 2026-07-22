#!/usr/bin/env python3
"""Local-only Kimi snapshot planner.

This intentionally does not download weights. Point it at a local Hugging Face
snapshot after Kimi K3 is released and it will verify architecture fields,
inspect the safetensors index, and emit a conversion plan for the C streamer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERT_SUFFIXES = (
    "mlp.experts.{eid}.gate_proj.weight",
    "mlp.experts.{eid}.up_proj.weight",
    "mlp.experts.{eid}.down_proj.weight",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def text_config(cfg: dict) -> dict:
    if isinstance(cfg.get("text_config"), dict):
        return cfg["text_config"]
    return cfg


def require(c: dict, name: str):
    if name not in c:
        raise SystemExit(f"missing config field: {name}")
    return c[name]


def layer_is_sparse(c: dict, layer: int) -> bool:
    first_dense = int(c.get("first_k_dense_replace", 0))
    freq = int(c.get("moe_layer_freq", 1))
    return layer >= first_dense and ((layer - first_dense) % freq == 0)


def plan_snapshot(snap: Path) -> dict:
    cfg_path = snap / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{cfg_path} not found")
    raw_cfg = load_json(cfg_path)
    cfg = text_config(raw_cfg)

    n_layers = int(require(cfg, "num_hidden_layers"))
    n_experts = int(require(cfg, "n_routed_experts"))
    hidden = int(require(cfg, "hidden_size"))
    inter = int(require(cfg, "moe_intermediate_size"))
    topk = int(require(cfg, "num_experts_per_tok"))

    sparse_layers = [l for l in range(n_layers) if layer_is_sparse(cfg, l)]
    index_path = snap / "model.safetensors.index.json"
    index_present = index_path.exists()
    missing_examples: list[str] = []
    shard_count = None
    weight_map = {}
    if index_present:
        idx = load_json(index_path)
        weight_map = idx.get("weight_map", {})
        shard_count = len(set(weight_map.values()))
        sample_layers = sparse_layers[:2] + sparse_layers[-2:]
        sample_layers = sorted(set(sample_layers))
        sample_experts = sorted(set([0, min(1, n_experts - 1), n_experts - 1]))
        for layer in sample_layers:
            for eid in sample_experts:
                for suffix in EXPERT_SUFFIXES:
                    name = f"model.layers.{layer}." + suffix.format(eid=eid)
                    if name not in weight_map:
                        missing_examples.append(name)
                        if len(missing_examples) >= 12:
                            break
                if len(missing_examples) >= 12:
                    break
            if len(missing_examples) >= 12:
                break

    expert_q4_bytes = (
        2 * (inter * ((hidden + 1) // 2) + inter * 4)
        + (hidden * ((inter + 1) // 2) + hidden * 4)
    )
    total_routed = len(sparse_layers) * n_experts

    return {
        "snapshot": str(snap),
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "is_deepseek_v3_like": {
            "scoring_func_sigmoid": cfg.get("scoring_func") == "sigmoid",
            "topk_method_noaux_tc": cfg.get("topk_method") == "noaux_tc",
            "n_group": cfg.get("n_group"),
            "topk_group": cfg.get("topk_group"),
            "has_mla_fields": all(k in cfg for k in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")),
        },
        "shape": {
            "num_hidden_layers": n_layers,
            "sparse_layers": len(sparse_layers),
            "n_routed_experts": n_experts,
            "total_routed_experts": total_routed,
            "num_experts_per_tok": topk,
            "hidden_size": hidden,
            "moe_intermediate_size": inter,
            "estimated_q4_expert_bytes": expert_q4_bytes,
            "estimated_q4_expert_store_gb": total_routed * expert_q4_bytes / 1e9,
            "cold_q4_read_per_token_gb": len(sparse_layers) * topk * expert_q4_bytes / 1e9,
        },
        "weights": {
            "index_present": index_present,
            "shard_count": shard_count,
            "missing_sample_expert_tensors": missing_examples,
        },
        "next_gate": "ready_for_tensor_mapping" if index_present and not missing_examples else "need_snapshot_or_tensor_names",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="local HF snapshot directory")
    ap.add_argument("--out", default=None, help="optional JSON plan output")
    args = ap.parse_args()

    plan = plan_snapshot(Path(args.snapshot))
    text = json.dumps(plan, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
