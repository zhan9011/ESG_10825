from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from esg.config import LABELS, TASKS, TASK_WEIGHTS
from esg.data import combined_labeled, read_data, seed_all
from esg.modeling import expand_binary


ROOT = Path("experiments/semantic_multitask")
MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
TASK_LABELS = {
    **LABELS,
    "evidence_quality": ["Clear", "Not Clear"],
}


def common_text(data: pd.DataFrame) -> list[str]:
    """Build the shared text template used by the semantic multitask model."""
    return [
        f"公司：{row.company} [SEP] 文本：{row.data}"
        for row in data.itertuples(index=False)
    ]


def encoded_labels(data: pd.DataFrame) -> np.ndarray:
    """Encode task labels, using -100 for rows ignored by a task head."""
    values = np.full((len(data), len(TASKS)), -100, dtype=np.int64)
    for column, task in enumerate(TASKS):
        mapping = {label: index for index, label in enumerate(TASK_LABELS[task])}
        values[:, column] = data[task].map(mapping).fillna(-100).astype(int)
    return values


class EncodedDataset:
    """Tokenized input dataset for multitask training and prediction."""

    def __init__(self, encoded: dict[str, np.ndarray], labels: np.ndarray | None):
        self.encoded = encoded
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encoded["input_ids"])

    def __getitem__(self, index: int) -> dict:
        import torch

        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encoded.items()
        }
        if self.labels is not None:
            item["task_labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class SemanticMultitaskModel:
    """Shared encoder with one classification head per ESG task."""

    def __new__(cls, pretrained_model: str, gradient_checkpointing: bool):
        import torch
        from transformers import AutoModel

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(pretrained_model)
                if gradient_checkpointing:
                    self.encoder.gradient_checkpointing_enable()
                hidden = self.encoder.config.hidden_size
                dropout = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
                self.dropout = torch.nn.Dropout(dropout)
                self.heads = torch.nn.ModuleDict(
                    {
                        task: torch.nn.Linear(hidden, len(TASK_LABELS[task]))
                        for task in TASKS
                    }
                )

            def forward(self, **inputs):
                hidden = self.encoder(**inputs).last_hidden_state[:, 0]
                hidden = self.dropout(hidden)
                return {task: head(hidden) for task, head in self.heads.items()}

        return Model()


@dataclass(frozen=True)
class Settings:
    epochs: int
    max_length: int
    batch_size: int
    eval_batch_size: int
    accumulation_steps: int
    learning_rate: float
    seed: int
    gradient_checkpointing: bool


def encode(tokenizer, data: pd.DataFrame, max_length: int) -> dict[str, np.ndarray]:
    """Tokenize rows with the same template for training and inference."""
    values = tokenizer(
        common_text(data),
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    return {key: value.astype(np.int64) for key, value in values.items()}


def class_weights(labels: np.ndarray, device, max_weight: float = 4.0) -> dict:
    """Compute sqrt-balanced class weights capped by max_weight."""
    import torch

    weights = {}
    for column, task in enumerate(TASKS):
        valid = labels[:, column][labels[:, column] >= 0]
        counts = np.bincount(valid, minlength=len(TASK_LABELS[task]))
        weights[task] = torch.tensor(
            [
                min(max_weight, math.sqrt(len(valid) / (len(counts) * max(1, count))))
                for count in counts
            ],
            dtype=torch.float32,
            device=device,
        )
    return weights


def predict(model, encoded: dict[str, np.ndarray], batch_size: int, device) -> dict:
    """Run batched model inference and return task probability arrays."""
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(EncodedDataset(encoded, None), batch_size=batch_size)
    output = {task: [] for task in TASKS}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**batch)
            for task in TASKS:
                output[task].append(torch.softmax(logits[task].float(), dim=-1).cpu())
    probabilities = {task: torch.cat(parts).numpy() for task, parts in output.items()}
    probabilities["evidence_quality"] = expand_binary(
        probabilities["evidence_quality"]
    )
    return probabilities


def train(
    train_data: pd.DataFrame,
    target_data: pd.DataFrame,
    settings: Settings,
) -> dict[str, np.ndarray]:
    """Train the semantic multitask model and predict target probabilities."""
    import torch
    import torch.nn.functional as functional
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    seed_all(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encoded = encode(tokenizer, train_data, settings.max_length)
    target_encoded = encode(tokenizer, target_data, settings.max_length)
    labels = encoded_labels(train_data)
    loader = DataLoader(
        EncodedDataset(train_encoded, labels),
        batch_size=settings.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    model = SemanticMultitaskModel(
        MODEL_NAME,
        settings.gradient_checkpointing,
    ).to(device)
    weights = class_weights(labels, device)
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=0.01)
    updates = math.ceil(len(loader) / settings.accumulation_steps) * settings.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(updates * 0.1)),
        updates,
    )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, settings.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(loader, start=1):
            targets = batch.pop("task_labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**batch)
                loss = torch.zeros((), device=device)
                active_weight = 0.0
                for column, task in enumerate(TASKS):
                    mask = targets[:, column] >= 0
                    if mask.any():
                        loss = loss + TASK_WEIGHTS[task] * functional.cross_entropy(
                            logits[task][mask],
                            targets[mask, column],
                            weight=weights[task],
                        )
                        active_weight += TASK_WEIGHTS[task]
                loss = loss / active_weight / settings.accumulation_steps
            loss.backward()
            losses.append(loss.item() * settings.accumulation_steps)
            if step % settings.accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        print(
            f"semantic_multitask epoch={epoch}/{settings.epochs} "
            f"loss={np.mean(losses):.4f}",
            flush=True,
        )

    probabilities = predict(model, target_encoded, settings.eval_batch_size, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def save_probabilities(path: Path, probabilities: dict[str, np.ndarray]) -> None:
    """Save task probability arrays to a compressed archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **probabilities)


def train_submit(args: argparse.Namespace) -> None:
    """Train on all labeled rows and save target test probabilities."""
    data = combined_labeled(args.train, args.validation)
    test = read_data(args.test, labeled=False)
    probability = train(data, test, settings_from_args(args))
    save_probabilities(args.root / "submission" / "target.npz", probability)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Shared-encoder semantic multitask model.")
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("train-submit")
    command.add_argument("--train", type=Path, default=Path("Data/vpesg4k_train_1000 V1.csv"))
    command.add_argument("--validation", type=Path, default=Path("Data/vpesg4k_val_1000.csv"))
    command.add_argument("--test", type=Path, default=Path("Data/vpesg4k_test_2000.csv"))
    command.add_argument("--root", type=Path, default=ROOT)
    command.add_argument("--epochs", type=int, default=4)
    command.add_argument("--max-length", type=int, default=512)
    command.add_argument("--batch-size", type=int, default=2)
    command.add_argument("--eval-batch-size", type=int, default=8)
    command.add_argument("--accumulation-steps", type=int, default=4)
    command.add_argument("--learning-rate", type=float, default=1e-5)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--gradient-checkpointing", action="store_true")
    command.set_defaults(func=train_submit)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.func(arguments)


if __name__ == "__main__":
    main()
