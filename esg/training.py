from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from esg.config import LABELS
from esg.data import applicable_mask, seed_all, task_text
from esg.modeling import dataset, encode, predict


@dataclass(frozen=True)
class TrainSettings:
    epochs: int
    max_length: int = 512
    batch_size: int = 2
    eval_batch_size: int = 8
    accumulation_steps: int = 4
    learning_rate: float = 1e-5
    max_class_weight: float = 4.0
    seed: int = 42
    gradient_checkpointing: bool = True


def train_classifier(
    train: pd.DataFrame,
    target: pd.DataFrame,
    task: str,
    pretrained_model: str,
    settings: TrainSettings,
    class_labels: list[str] | None = None,
    validation: pd.DataFrame | None = None,
) -> tuple[np.ndarray | None, np.ndarray, dict]:
    import torch
    import torch.nn.functional as functional
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_cosine_schedule_with_warmup

    seed_all(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = class_labels or LABELS[task]
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_mask = applicable_mask(train, task) & train[task].isin(labels)
    train_subset = train.loc[train_mask].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    x_train = encode(tokenizer, task_text(train_subset, task), settings.max_length)
    x_target = encode(tokenizer, task_text(target, task), settings.max_length)
    x_validation = (
        encode(tokenizer, task_text(validation, task), settings.max_length)
        if validation is not None
        else None
    )
    y_train = train_subset[task].map(label_to_id).to_numpy(dtype=np.int64)
    loader = DataLoader(
        dataset(x_train, y_train),
        batch_size=settings.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    if settings.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    counts = train_subset[task].value_counts()
    weights = torch.tensor(
        [
            min(
                settings.max_class_weight,
                np.sqrt(
                    len(train_subset)
                    / (len(labels) * max(1, counts.get(label, 0)))
                ),
            )
            for label in labels
        ],
        dtype=torch.float32,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=0.01)
    updates = int(np.ceil(len(loader) / settings.accumulation_steps)) * settings.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(updates * 0.1)),
        updates,
    )

    best_score = -1.0
    best_epoch = settings.epochs
    best_state = None
    best_validation = None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, settings.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(loader, start=1):
            targets = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**batch).logits
                loss = (
                    functional.cross_entropy(logits, targets, weight=weights)
                    / settings.accumulation_steps
                )
            loss.backward()
            losses.append(loss.item() * settings.accumulation_steps)
            if step % settings.accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        message = (
            f"{task} epoch={epoch}/{settings.epochs} loss={np.mean(losses):.4f}"
        )
        if validation is not None and x_validation is not None:
            probability = predict(model, x_validation, settings.eval_batch_size, device)
            mask = applicable_mask(validation, task) & validation[task].isin(labels)
            guessed = np.asarray(labels)[probability[mask.to_numpy()].argmax(axis=1)]
            score = f1_score(
                validation.loc[mask, task],
                guessed,
                labels=LABELS[task],
                average="macro",
                zero_division=0,
            )
            message += f" macro_f1={score:.4f}"
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_validation = probability
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        print(message)

    if best_state is not None:
        model.load_state_dict(best_state)
    target_probability = predict(model, x_target, settings.eval_batch_size, device)
    metrics = {"best_epoch": best_epoch}
    if validation is not None:
        metrics["macro_f1"] = best_score
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_validation, target_probability, metrics
