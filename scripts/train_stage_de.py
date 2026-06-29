#!/usr/bin/env python3
"""Wrapper for Qwen Stage D + Stage E training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import main as train_main


def _inject_defaults(argv: list[str]) -> list[str]:
    result = [argv[0]]
    if "--config" not in argv:
        result.extend(["--config", str(ROOT / "configs" / "swdd_qwen35_stage_de_fullparam.yaml")])
    result.extend(argv[1:])
    return result


if __name__ == "__main__":
    sys.argv = _inject_defaults(sys.argv)
    train_main()
