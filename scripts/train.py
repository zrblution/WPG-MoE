#!/usr/bin/env python3
"""Train the public WPG-MoE algorithm package.

This entrypoint assumes that user-level JSONL samples have already been
prepared. It keeps the algorithmic training path used by WPG-MoE: Stage-D
expert warm start followed by Stage-E joint training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.full_model import WPGMoEModel
from src.training.distributed import barrier, broadcast_object, is_distributed, is_main_process
from src.training.joint_trainer import train_joint
from src.training.warm_start import warm_start_experts
from src.utils.config import load_yaml_config, resolve_path
from src.utils.io_utils import ensure_dir


def _load_train_samples(train_path: Path) -> list[dict]:
    rows = []
    with train_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _resolve_requested_device(device_value: str | None) -> str:
    if not device_value:
        return "cuda" if torch.cuda.is_available() else "cpu"
    normalized = device_value.strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        if is_main_process():
            print(f"[Train] requested device '{device_value}' is unavailable; falling back to cpu", flush=True)
        return "cpu"
    return device_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_path")
    parser.add_argument("--val_path")
    parser.add_argument("--device")
    parser.add_argument("--skip_stage_d", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.train_path:
        config["train_path"] = args.train_path
    if args.val_path:
        config["val_path"] = args.val_path
    if args.device:
        config["device"] = args.device
    config["deepspeed_enabled"] = False

    train_path = resolve_path(config["train_path"])
    val_path = resolve_path(config["val_path"])
    warmstart_save_path = resolve_path(config["warmstart_save_path"])
    final_save_path = resolve_path(config["save_path"])
    log_path = resolve_path(config["log_path"])
    if train_path is None or val_path is None:
        raise ValueError("train_path and val_path are required")
    if warmstart_save_path is None or final_save_path is None or log_path is None:
        raise ValueError("warmstart_save_path, save_path, and log_path are required")

    ensure_dir(warmstart_save_path.parent)
    ensure_dir(final_save_path.parent)
    ensure_dir(log_path.parent)

    resolved_device = _resolve_requested_device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    config["device"] = resolved_device
    device = torch.device(resolved_device)

    model = WPGMoEModel(config).to(device)
    dropout_disable_stats = getattr(model, "dropout_disable_stats", {"disabled": 0, "already_zero": 0, "total": 0})
    config["dropout_disable_stats"] = dropout_disable_stats

    train_samples = _load_train_samples(train_path)
    warm_start_history: dict
    if args.skip_stage_d:
        if is_main_process():
            print("[Train] skip Stage D", flush=True)
        warm_start_history = {"skipped": True, "loaded_from": None}
    else:
        if is_main_process():
            print(f"[Train] Stage D start: train_samples={len(train_samples)}", flush=True)
        warm_start_history = warm_start_experts(model, train_samples, config)
        if is_main_process():
            torch.save(model.state_dict(), warmstart_save_path)
            print(f"[Train] Stage D complete: saved {warmstart_save_path}", flush=True)
        barrier()
        warm_start_history = broadcast_object(warm_start_history if is_main_process() else None, src=0)
        if is_distributed() and not is_main_process():
            state_dict = torch.load(warmstart_save_path, map_location=device)
            model.load_state_dict(state_dict)
        barrier()

    train_config = dict(config)
    train_config["save_path"] = str(final_save_path)
    train_config["log_path"] = str(log_path)
    if is_main_process():
        print(f"[Train] Stage E start: train={train_path} val={val_path}", flush=True)
    joint_history = train_joint(model, str(train_path), str(val_path), train_config)
    if is_main_process():
        print(f"[Train] Stage E complete: best_f1={joint_history['best_f1']:.4f}", flush=True)
        print(
            json.dumps(
                {
                    "dropout_disable_stats": dropout_disable_stats,
                    "stage_d": warm_start_history,
                    "stage_e": joint_history,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
