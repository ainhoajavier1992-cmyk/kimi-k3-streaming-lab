#!/usr/bin/env python3
"""Probe public Kimi metadata without downloading model weights."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


K3_REPOS = [
    "moonshotai/Kimi-K3",
    "reteetzad/Kimi-K3",
    "audnai/penclaw-Kimi-K3.0-abliterated-GGUF",
]
K2_REPOS = [
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.6",
]


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "k3stream-probe/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def try_fetch_json(url: str):
    try:
        return fetch_json(url)
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "url": url}
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic tool
        return {"error": str(exc), "url": url}


def repo_api(repo: str):
    return try_fetch_json(f"https://huggingface.co/api/models/{repo}")


def repo_tree(repo: str):
    return try_fetch_json(f"https://huggingface.co/api/models/{repo}/tree/main")


def raw_json(repo: str, name: str):
    return try_fetch_json(f"https://huggingface.co/{repo}/raw/main/{name}")


def text_config(cfg: dict) -> dict:
    if isinstance(cfg.get("text_config"), dict):
        return cfg["text_config"]
    return cfg


def summarize_config(repo: str, cfg: dict) -> dict:
    c = text_config(cfg)
    keys = [
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "first_k_dense_replace",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "intermediate_size",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "n_group",
        "topk_group",
        "topk_method",
        "scoring_func",
        "norm_topk_prob",
        "routed_scaling_factor",
        "max_position_embeddings",
        "quantization_config",
    ]
    return {"repo": repo, **{k: c.get(k) for k in keys if k in c}}


def main() -> int:
    print("Kimi K3 public release probe")
    found_usable_k3 = False
    for repo in K3_REPOS:
        k3 = repo_api(repo)
        tree = repo_tree(repo)
        siblings = [s.get("rfilename") for s in k3.get("siblings", []) if isinstance(s, dict)]
        tree_paths = [s.get("path") for s in tree if isinstance(s, dict)] if isinstance(tree, list) else []
        names = sorted(set(siblings + tree_paths))
        has_config = "config.json" in names
        has_weights = any(str(s).endswith(".safetensors") or str(s).endswith(".gguf") for s in names)
        found_usable_k3 = found_usable_k3 or (has_config and has_weights)
        print(json.dumps(
            {
                "candidate_repo": repo,
                "api_error": k3.get("error") if isinstance(k3, dict) else None,
                "tree_error": tree.get("error") if isinstance(tree, dict) else None,
                "lastModified": k3.get("lastModified") if isinstance(k3, dict) else None,
                "usedStorage": k3.get("usedStorage") if isinstance(k3, dict) else None,
                "files": names[:40],
                "has_config": has_config,
                "has_weights": has_weights,
                "cardData": k3.get("cardData") if isinstance(k3, dict) else None,
            },
            indent=2,
        ))

    print("\nKimi K3 official public summary")
    print(json.dumps(
        {
            "api_model_id": "kimi-k3",
            "public_summary": {
                "total_parameters": "2.8T",
                "context_window": "1M tokens",
                "attention": "Kimi Delta Attention",
                "residual": "Attention Residuals",
                "moe": "Stable LatentMoE, 16 active out of 896 experts",
                "modalities": "text + native vision",
            },
            "source_docs": [
                "https://platform.kimi.ai/docs/guide/kimi-k3-quickstart",
                "https://platform.kimi.ai/docs/models.md",
                "https://www.kimi.com/blog/kimi-k3",
            ],
            "local_runtime_status": "High-level architecture is public; exact config/tokenizer/weight-index/modeling files are still required.",
        },
        indent=2,
    ))

    print("\nKimi precedent configs")
    for repo in K2_REPOS:
        cfg = raw_json(repo, "config.json")
        if "error" in cfg:
            print(json.dumps({"repo": repo, "error": cfg["error"]}, indent=2))
            continue
        print(json.dumps(summarize_config(repo, cfg), indent=2))

    if not found_usable_k3:
        print("\nSTATUS: No candidate K3 repo currently exposes both config and weights. Continue synthetic engine work; do not download model weights yet.")
    else:
        print("\nSTATUS: A candidate K3 repo exposes config and weights. Next step: run tools/kimi_convert_skeleton.py against a local snapshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
