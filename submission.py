from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from esg.config import ENSEMBLE_WEIGHTS, TASKS
from esg.data import probabilities_to_prediction, read_data, save_submission


def load_ensemble(root: Path, array_key: str) -> dict[str, np.ndarray]:
    probabilities = {}
    for task in TASKS:
        parts = []
        for model_key, weight in ENSEMBLE_WEIGHTS[task].items():
            path = root / model_key / f"{task}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing artifact: {path}")
            parts.append(weight * np.load(path, allow_pickle=False)[array_key])
        probabilities[task] = sum(parts)
    return probabilities


def blend_branches(
    full: dict[str, np.ndarray],
    train_only: dict[str, np.ndarray],
    full_weight: float,
) -> dict[str, np.ndarray]:
    return {
        task: full_weight * full[task] + (1 - full_weight) * train_only[task]
        for task in TASKS
    }


def main(args: argparse.Namespace) -> None:
    full = load_ensemble(args.full_root, "target")
    probabilities = full
    if args.full_weight < 1:
        train_only = load_ensemble(args.train_only_root, "target")
        probabilities = blend_branches(full, train_only, args.full_weight)
    test = read_data(args.test, labeled=False)
    save_submission(test, probabilities_to_prediction(probabilities), args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a submission from model probabilities.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--train-only-root", type=Path)
    parser.add_argument("--full-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.full_weight <= 1:
        raise ValueError("--full-weight must be between 0 and 1")
    if args.full_weight < 1 and args.train_only_root is None:
        raise ValueError("--train-only-root is required when --full-weight is below 1")
    main(args)
