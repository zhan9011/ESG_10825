from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.config import load_config, set_config_value
from src.common.logging_utils import configure_logging
from src.training import pipeline


def parse_value(raw: str) -> Any:
    """Parse simple command-line override values."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Train promise verification workflow.")
    root.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    root.add_argument(
        "--stage",
        choices=[
            "all",
            "prepare-baseline",
            "semantic-multitask",
            "semantic-bge",
            "lexical-tfidf",
        ],
        default="all",
    )
    root.add_argument("--force", action="store_true")
    root.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, for example training.batch_size=4.",
    )
    return root


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set value: {item}")
        key, value = item.split("=", 1)
        set_config_value(config, key, parse_value(value))
    return config


def main() -> None:
    args = parser().parse_args()
    config = load_runtime_config(args)
    configure_logging(config["runtime"].get("log_level", "INFO"))

    if args.stage == "all":
        pipeline.run_all(config, force=args.force)
    elif args.stage == "prepare-baseline":
        pipeline.prepare_baseline(config, force=args.force)
    elif args.stage == "semantic-multitask":
        pipeline.train_semantic_multitask(config, force=args.force)
    elif args.stage == "semantic-bge":
        pipeline.train_semantic_bge(config, force=args.force)
    elif args.stage == "lexical-tfidf":
        pipeline.train_lexical_tfidf(config, force=args.force)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
