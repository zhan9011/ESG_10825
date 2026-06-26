from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from esg.config import LABELS, TASKS
from esg.data import (
    applicable_mask,
    combined_labeled,
    probabilities_to_prediction,
    read_data,
    save_submission,
)
from submission import blend_branches, load_ensemble


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_path(config: dict[str, Any], section: str, key: str) -> Path:
    return project_path(config[section][key])


def run_command(command: list[str], *, dry_run: bool = False) -> None:
    LOGGER.info("Running: %s", subprocess.list2cmdline(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def texts(data: pd.DataFrame) -> list[str]:
    """Build the shared company/text representation for linear extra models."""
    return [f"{row.company} [SEP] {row.data}" for row in data.itertuples(index=False)]


def copy_tree(source: Path, target: Path) -> None:
    """Copy a generated artifact tree if the source exists."""
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def prepare_baseline(
    config: dict[str, Any],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Generate or refresh the dual-source baseline prediction artifacts."""
    source_predictions = config_path(config, "paths", "source_predictions")
    baseline_predictions = config_path(config, "paths", "baseline_predictions")

    if force or not (source_predictions / "full").exists():
        output = PROJECT_ROOT / "outputs" / "dual_source_baseline_retrained.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "dual_source_baseline_pipeline.py",
            "--output",
            str(output),
        ]
        if force:
            command.append("--force")
        run_command(command, dry_run=dry_run)
    else:
        LOGGER.info("Reuse %s", source_predictions)

    copy_tree(source_predictions, baseline_predictions)


def baseline_root(config: dict[str, Any]) -> Path:
    preferred = config_path(config, "paths", "baseline_predictions")
    if preferred.exists():
        return preferred
    return config_path(config, "paths", "source_predictions")


def baseline_probabilities(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load the dual-source baseline target probability blend."""
    root = baseline_root(config)
    full = load_ensemble(root / "full", "target")
    train_only = load_ensemble(root / "train_only", "target")
    return blend_branches(full, train_only, float(config["blend"]["baseline_full_weight"]))


def load_archive(path: Path) -> dict[str, np.ndarray]:
    values = np.load(path, allow_pickle=False)
    return {task: values[task] for task in TASKS if task in values.files}


def save_archive(path: Path, values: dict[str, np.ndarray]) -> None:
    """Write a compressed probability archive with exactly the provided keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def lexical_specs(config: dict[str, Any]) -> dict[str, tuple[str, tuple[int, int]]]:
    specs = config["lexical_tfidf"]
    return {
        "evidence_status": (
            specs["evidence_status"]["model_name"],
            (
                int(specs["evidence_status"]["ngram_min"]),
                int(specs["evidence_status"]["ngram_max"]),
            ),
        ),
        "verification_timeline": (
            specs["verification_timeline"]["model_name"],
            (
                int(specs["verification_timeline"]["ngram_min"]),
                int(specs["verification_timeline"]["ngram_max"]),
            ),
        ),
    }


def train_lexical_tfidf(config: dict[str, Any], *, force: bool = False) -> None:
    """Train TF-IDF LinearSVC extra models and save target probability archives."""
    root = config_path(config, "paths", "lexical_tfidf")
    train = combined_labeled(
        config_path(config, "data", "train"),
        config_path(config, "data", "validation"),
    )
    test = read_data(config_path(config, "data", "test"), labeled=False)
    train_text = np.asarray(texts(train))
    test_text = texts(test)

    for task, (name, ngrams) in lexical_specs(config).items():
        target_output = root / "target" / name / f"{task}.npz"
        if target_output.exists() and not force:
            LOGGER.info("Reuse lexical TF-IDF artifact for %s", task)
            continue

        mask = applicable_mask(train, task).to_numpy()
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngrams,
            min_df=2,
            max_df=0.995,
            sublinear_tf=True,
            max_features=250000,
            dtype=np.float32,
        )
        x_train = vectorizer.fit_transform(train_text[mask])
        x_test = vectorizer.transform(test_text)
        classifier = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000)
        classifier.fit(x_train, train.loc[mask, task])
        decision = classifier.decision_function(x_test)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        raw = softmax(decision, axis=1)
        target_probability = np.zeros((len(test), len(LABELS[task])), dtype=np.float32)
        for index, label in enumerate(LABELS[task]):
            target_probability[:, index] = raw[
                :, np.where(classifier.classes_ == label)[0][0]
            ]
        save_archive(target_output, {"target": target_probability})
        LOGGER.info("Saved %s", target_output)


def train_semantic_bge(config: dict[str, Any], *, force: bool = False) -> None:
    """Train BGE embedding logistic regression extra model for promise_status."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    root = config_path(config, "paths", "semantic_bge")
    embeddings_path = root / "embeddings.npy"
    output = root / "logreg_c10" / "promise_status.npz"
    train = combined_labeled(
        config_path(config, "data", "train"),
        config_path(config, "data", "validation"),
    )
    test = read_data(config_path(config, "data", "test"), labeled=False)
    if not force and output.exists():
        try:
            existing = np.load(output, allow_pickle=False)
            if "target" in existing.files:
                LOGGER.info("Reuse %s", output)
                return
        except Exception:
            LOGGER.warning("Ignoring unreadable archive before BGE refresh: %s", output)

    if force or not embeddings_path.exists():
        all_data = pd.concat([train, test], ignore_index=True)
        model_name = config["semantic_bge"]["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        parts = []
        all_text = texts(all_data)
        for start in range(0, len(all_text), 8):
            batch = tokenizer(
                all_text[start : start + 8],
                max_length=int(config["training"]["max_length"]),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                hidden = model(**batch).last_hidden_state
            attention = batch["attention_mask"].unsqueeze(-1)
            mean = (hidden * attention).sum(1) / attention.sum(1)
            dense = torch.cat([hidden[:, 0], mean], dim=1)
            dense = torch.nn.functional.normalize(dense.float(), dim=1)
            parts.append(dense.cpu().numpy().astype(np.float16))
        root.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, np.concatenate(parts))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    embeddings = np.load(embeddings_path).astype(np.float32)
    train_x = embeddings[: len(train)]
    test_x = embeddings[len(train) :]
    task = "promise_status"
    classifier = LogisticRegression(
        C=float(config["semantic_bge"]["logistic_c"]),
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
    )
    classifier.fit(train_x, train[task])
    raw = classifier.predict_proba(test_x)
    target_probability = np.zeros((len(test), len(LABELS[task])), dtype=np.float32)
    for index, label in enumerate(LABELS[task]):
        target_probability[:, index] = raw[
            :, np.where(classifier.classes_ == label)[0][0]
        ]

    save_archive(output, {"target": target_probability})
    LOGGER.info("Saved %s", output)


def train_semantic_multitask(
    config: dict[str, Any],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Train semantic multitask target probability archive."""
    root = config_path(config, "paths", "semantic_multitask")
    target_archive = root / "submission" / "target.npz"
    if target_archive.exists() and not force:
        LOGGER.info("Reuse %s", target_archive)
        return

    command = [
        sys.executable,
        "-m",
        "src.training.semantic_multitask",
        "train-submit",
        "--train",
        str(config_path(config, "data", "train")),
        "--validation",
        str(config_path(config, "data", "validation")),
        "--test",
        str(config_path(config, "data", "test")),
        "--root",
        str(root),
        "--epochs",
        str(config["training"]["semantic_multitask_epochs"]),
        "--max-length",
        str(config["training"]["max_length"]),
        "--batch-size",
        str(config["training"]["batch_size"]),
        "--eval-batch-size",
        str(config["training"]["eval_batch_size"]),
        "--accumulation-steps",
        str(config["training"]["accumulation_steps"]),
        "--learning-rate",
        str(config["training"]["learning_rate"]),
        "--seed",
        str(config["runtime"]["seed"]),
    ]
    if bool(config["training"]["gradient_checkpointing"]):
        command.append("--gradient-checkpointing")
    run_command(command, dry_run=dry_run)


def components(config: dict[str, Any]) -> tuple[dict, dict, dict]:
    """Load baseline, semantic multitask, and extra component target probabilities."""
    semantic_root = config_path(config, "paths", "semantic_multitask")
    bge_root = config_path(config, "paths", "semantic_bge")
    lexical_root = config_path(config, "paths", "lexical_tfidf")
    specs = lexical_specs(config)
    baseline = baseline_probabilities(config)
    semantic = load_archive(semantic_root / "submission" / "target.npz")
    extra = {
        "promise_status": np.load(
            bge_root / "logreg_c10/promise_status.npz",
            allow_pickle=False,
        )["target"],
        "evidence_status": np.load(
            lexical_root / "target" / specs["evidence_status"][0] / "evidence_status.npz",
            allow_pickle=False,
        )["target"],
        "verification_timeline": np.load(
            lexical_root
            / "target"
            / specs["verification_timeline"][0]
            / "verification_timeline.npz",
            allow_pickle=False,
        )["target"],
    }
    return baseline, semantic, extra


def blended_probabilities(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Apply the fixed promise verification recipe to component probabilities."""
    baseline, semantic, extra = components(config)
    probabilities = {}
    for task in TASKS:
        weight = config["blend"][task]
        probabilities[task] = (
            float(weight["baseline"]) * baseline[task]
            + float(weight["semantic_multitask"]) * semantic[task]
        )
        if float(weight["extra"]):
            probabilities[task] += float(weight["extra"]) * extra[task]
    return probabilities


def build_submission(config: dict[str, Any], output: Path | None = None) -> Path:
    """Build the final submission CSV from the fixed recipe."""
    output_path = project_path(output or config["paths"]["output"])
    test = read_data(config_path(config, "data", "test"), labeled=False)
    save_submission(
        test,
        probabilities_to_prediction(blended_probabilities(config)),
        output_path,
    )
    return output_path


def run_all(config: dict[str, Any], *, force: bool = False) -> Path:
    """Run every submission training stage and build a submission."""
    prepare_baseline(config, force=force)
    train_semantic_multitask(config, force=force)
    train_semantic_bge(config, force=force)
    train_lexical_tfidf(config, force=force)
    return build_submission(config)
