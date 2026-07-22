#!/usr/bin/env python3
"""Probe public Kimi metadata without downloading model weights."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


K3_REPO = "reteetzad/Kimi-K3"
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
    k3 = repo_api(K3_REPO)
    siblings = [s.get("rfilename") for s in k3.get("siblings", []) if isinstance(s, dict)]
    print(json.dumps(
        {
            "k3_repo": K3_REPO,
            "lastModified": k3.get("lastModified"),
            "usedStorage": k3.get("usedStorage"),
            "siblings": siblings,
            "has_config": "config.json" in siblings,
            "has_weights": any(str(s).endswith(".safetensors") for s in siblings),
            "cardData": k3.get("cardData"),
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

    if "config.json" not in siblings:
        print("\nSTATUS: K3 config/weights are not published yet. Continue synthetic engine work; do not download model weights yet.")
    else:
        print("\nSTATUS: K3 config is present. Next step: run tools/kimi_convert_skeleton.py against a local snapshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
