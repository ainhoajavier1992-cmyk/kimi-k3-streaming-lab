#!/usr/bin/env python3
"""Smoke/regression tests for the Kimi K3 streaming scaffold."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "k3stream"


def run(args: list[str], cwd: Path = ROOT) -> str:
    p = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise AssertionError(f"command failed: {' '.join(args)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.stdout


def checksums(output: str) -> list[str]:
    return re.findall(r"checksum=([-0-9.]+)", output)


def stats(output: str) -> dict[str, float]:
    m = re.search(r"stats requests=(\d+) hits=(\d+) misses=(\d+) hit_rate=([0-9.]+)% bytes_read=([0-9.]+)", output)
    if not m:
        raise AssertionError(f"missing stats line:\n{output}")
    return {
        "requests": int(m.group(1)),
        "hits": int(m.group(2)),
        "misses": int(m.group(3)),
        "hit_rate": float(m.group(4)),
        "bytes_read": float(m.group(5)),
    }


def test_cache_changes_io_not_math(tmp: Path) -> None:
    model = tmp / "tiny"
    run([
        str(BIN),
        "fixture",
        "--out",
        str(model),
        "--layers",
        "5",
        "--experts",
        "20",
        "--hidden",
        "48",
        "--inter",
        "96",
        "--topk",
        "3",
        "--seed",
        "19",
    ])
    cold = run([str(BIN), "run", "--model", str(model), "--tokens", "16", "--cache", "1"])
    stats_file = tmp / "usage.txt"
    warm = run([str(BIN), "run", "--model", str(model), "--tokens", "16", "--cache", "8", "--stats-out", str(stats_file)])
    if checksums(cold) != checksums(warm):
        raise AssertionError("cache size changed math output")
    pinned = run([
        str(BIN),
        "run",
        "--model",
        str(model),
        "--tokens",
        "16",
        "--cache",
        "8",
        "--pin",
        str(stats_file),
        "--pin-per-layer",
        "3",
    ])
    if checksums(warm) != checksums(pinned):
        raise AssertionError("pinning changed math output")
    batched = run([str(BIN), "run", "--model", str(model), "--tokens", "16", "--cache", "8", "--batch", "4"])
    if checksums(warm) != checksums(batched):
        raise AssertionError("batch-union changed math output")
    cold_stats = stats(cold)
    warm_stats = stats(warm)
    if warm_stats["misses"] >= cold_stats["misses"]:
        raise AssertionError(f"larger cache did not reduce misses: cold={cold_stats}, warm={warm_stats}")
    pinned_stats = stats(pinned)
    if pinned_stats["misses"] > warm_stats["misses"]:
        raise AssertionError(f"pinning increased misses: warm={warm_stats}, pinned={pinned_stats}")


def test_convert_skeleton_fake_snapshot(tmp: Path) -> None:
    snap = tmp / "snapshot"
    snap.mkdir()
    cfg = {
        "model_type": "kimi_k2",
        "architectures": ["DeepseekV3ForCausalLM"],
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": 6,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "intermediate_size": 64,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
    }
    (snap / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    weight_map = {}
    for layer in range(1, 4):
        for eid in range(6):
            for suffix in ("gate_proj", "up_proj", "down_proj"):
                weight_map[f"model.layers.{layer}.mlp.experts.{eid}.{suffix}.weight"] = f"model-{layer}.safetensors"
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    out = run(["python3", "tools/kimi_convert_skeleton.py", "--snapshot", str(snap)])
    plan = json.loads(out)
    if plan["next_gate"] != "ready_for_tensor_mapping":
        raise AssertionError(plan)
    if plan["shape"]["sparse_layers"] != 3:
        raise AssertionError(plan)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="k3stream-test-") as td:
        tmp = Path(td)
        test_cache_changes_io_not_math(tmp)
        test_convert_skeleton_fake_snapshot(tmp)
    print("tests passed")
    return 0


if __name__ == "__main__":
    if not BIN.exists():
        subprocess.run(["make"], cwd=ROOT, check=True)
    raise SystemExit(main())
