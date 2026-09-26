"""Shared argparse setup for every pipeline script: --mock, --limit N, --config PATH."""
from __future__ import annotations

import argparse

from common.config import DEFAULT_CONFIG_PATH, Config, load_config


def base_parser(description: str | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--mock", action="store_true", help="run on mock artifacts (fast, no downloads)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N base images (tiny pipeline)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return ap


def parse(ap: argparse.ArgumentParser) -> tuple[argparse.Namespace, Config]:
    args = ap.parse_args()
    return args, load_config(args.config)


def not_implemented(step: str, owner: str) -> None:
    """Uniform message so `make pipeline` tells you exactly which step is still a stub."""
    raise SystemExit(f"[{step}] not implemented yet — TODO({owner}). See TASKS.md.")
