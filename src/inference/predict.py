from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.config import load_config, set_config_value
from src.common.logging_utils import configure_logging
from src.inference.predictor import predict_submission


def parse_value(raw: str) -> Any:
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
    root = argparse.ArgumentParser(description="Run promise verification inference.")
    root.add_argument("--config", type=Path, default=Path("config/inference.yaml"))
    root.add_argument("--output", type=Path)
    root.add_argument(
        "--cache-policy",
        choices=["auto", "refresh", "off"],
        help="auto uses valid caches and recomputes invalid ones; refresh/off rebuild.",
    )
    root.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, for example paths.output=outputs/submission.csv.",
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
    output = predict_submission(
        config,
        output=args.output,
        cache_policy=args.cache_policy,
    )
    print(output)


if __name__ == "__main__":
    main()
