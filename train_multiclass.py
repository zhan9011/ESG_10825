from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from esg.config import LABELS
from esg.data import combined_labeled, read_data
from esg.training import TrainSettings, train_classifier


def main(args: argparse.Namespace) -> None:
    if args.branch == "full":
        train = combined_labeled(args.train, args.validation)
        validation = None
    else:
        train = read_data(args.train, labeled=True)
        validation = read_data(args.validation, labeled=True)
    target = read_data(args.target, labeled=False)
    settings = TrainSettings(
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    val_probability, target_probability, metrics = train_classifier(
        train=train,
        validation=validation,
        target=target,
        task=args.task,
        pretrained_model=args.pretrained_model,
        settings=settings,
    )
    output_dir = args.output_root / args.model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "target": target_probability,
        "labels": np.asarray(LABELS[args.task]),
    }
    if val_probability is not None:
        arrays["validation"] = val_probability
    np.savez_compressed(output_dir / f"{args.task}.npz", **arrays)
    (output_dir / f"{args.task}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one multiclass task model.")
    parser.add_argument("--branch", choices=["full", "train-only"], required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--pretrained-model", required=True)
    parser.add_argument("--task", choices=list(LABELS), required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    main(parser.parse_args())
