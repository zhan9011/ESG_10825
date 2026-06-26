from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from esg.config import MODEL_SPECS


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = Path("Data/vpesg4k_train_1000 V1.csv")
DEFAULT_VALIDATION = Path("Data/vpesg4k_val_1000.csv")
DEFAULT_TEST = Path("Data/vpesg4k_test_2000.csv")
PREDICTION_FULL = Path("experiments/predictions/full")
PREDICTION_TRAIN_ONLY = Path("experiments/predictions/train_only")


def run(command: list[str], dry_run: bool) -> None:
    print(f"\n> {subprocess.list2cmdline(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def common_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--max-length",
        str(args.max_length),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--accumulation-steps",
        str(args.accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--gradient-checkpointing",
    ]


def train_submission_models(args: argparse.Namespace) -> None:
    """Train or reuse full-data and train-only target prediction models."""
    for branch, output_root in [
        ("full", args.full_root),
        ("train-only", args.train_only_root),
    ]:
        print(f"\n=== {branch} submission models ===")
        for index, spec in enumerate(MODEL_SPECS, start=1):
            artifact = output_root / spec.model_key / f"{spec.task}.npz"
            if artifact.exists() and not args.force:
                print(f"[{index}/10] Reuse {artifact}")
                continue
            script = "train_binary.py" if spec.binary else "train_multiclass.py"
            command = [
                sys.executable,
                script,
                "--branch",
                branch,
                "--train",
                str(args.train),
                "--validation",
                str(args.validation),
                "--target",
                str(args.test),
                "--output-root",
                str(output_root),
                "--model-key",
                spec.model_key,
                "--pretrained-model",
                spec.pretrained_model,
                "--epochs",
                str(spec.epochs),
                *common_arguments(args),
            ]
            if not spec.binary:
                command.extend(["--task", spec.task])
            print(f"[{index}/10] {spec.model_key}/{spec.task}")
            run(command, args.dry_run)


def build_submission(args: argparse.Namespace, full_weight: float) -> None:
    run(
        [
            sys.executable,
            "submission.py",
            "--test",
            str(args.test),
            "--full-root",
            str(args.full_root),
            "--train-only-root",
            str(args.train_only_root),
            "--full-weight",
            str(full_weight),
            "--output",
            str(args.output),
        ],
        args.dry_run,
    )


def run_pipeline(args: argparse.Namespace, full_weight: float) -> None:
    train_submission_models(args)
    build_submission(args, full_weight)


def pipeline_parser(name: str, default_output: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{name} submission pipeline.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=Path(default_output))
    parser.add_argument("--full-root", type=Path, default=PREDICTION_FULL)
    parser.add_argument("--train-only-root", type=Path, default=PREDICTION_TRAIN_ONLY)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser
