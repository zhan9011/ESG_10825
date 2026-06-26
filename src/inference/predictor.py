from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from esg.config import ENSEMBLE_WEIGHTS, LABELS, TASKS
from esg.data import read_data
from src.training import pipeline


LOGGER = logging.getLogger(__name__)


class CacheValidationError(RuntimeError):
    """Raised when a probability cache is missing, stale, or unreadable."""


def _load_array(path: Path, key: str, rows: int, columns: int) -> np.ndarray:
    if not path.exists():
        raise CacheValidationError(f"Missing cache: {path}")
    try:
        values = np.load(path, allow_pickle=False)
        if key not in values.files:
            raise CacheValidationError(f"Missing key '{key}' in {path}")
        array = values[key]
    except CacheValidationError:
        raise
    except Exception as exc:
        raise CacheValidationError(f"Unreadable cache {path}: {exc}") from exc
    if array.shape != (rows, columns):
        raise CacheValidationError(
            f"Invalid shape for {path}:{key}: expected {(rows, columns)}, "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise CacheValidationError(f"Cache contains non-finite values: {path}:{key}")
    return array


def _test_rows(config: dict[str, Any]) -> int:
    test = read_data(pipeline.config_path(config, "data", "test"), labeled=False)
    return len(test)


def validate_baseline_cache(config: dict[str, Any]) -> None:
    rows = _test_rows(config)
    root = pipeline.baseline_root(config)
    for branch in ["full", "train_only"]:
        for task in TASKS:
            for model_key in ENSEMBLE_WEIGHTS[task]:
                path = root / branch / model_key / f"{task}.npz"
                _load_array(path, "target", rows, len(LABELS[task]))


def validate_semantic_multitask_cache(config: dict[str, Any]) -> None:
    rows = _test_rows(config)
    root = pipeline.config_path(config, "paths", "semantic_multitask")
    path = root / "submission" / "target.npz"
    for task in TASKS:
        _load_array(path, task, rows, len(LABELS[task]))


def validate_semantic_bge_cache(config: dict[str, Any]) -> None:
    rows = _test_rows(config)
    path = (
        pipeline.config_path(config, "paths", "semantic_bge")
        / "logreg_c10"
        / "promise_status.npz"
    )
    _load_array(path, "target", rows, len(LABELS["promise_status"]))


def validate_lexical_tfidf_cache(config: dict[str, Any]) -> None:
    rows = _test_rows(config)
    root = pipeline.config_path(config, "paths", "lexical_tfidf")
    specs = pipeline.lexical_specs(config)
    for task in ["evidence_status", "verification_timeline"]:
        path = root / "target" / specs[task][0] / f"{task}.npz"
        _load_array(path, "target", rows, len(LABELS[task]))


def ensure_prediction_artifacts(config: dict[str, Any], cache_policy: str = "auto") -> None:
    """Ensure prediction artifacts exist; refresh only when required."""
    if cache_policy not in {"auto", "refresh", "off"}:
        raise ValueError(f"Unknown cache policy: {cache_policy}")

    refresh_all = cache_policy in {"refresh", "off"}
    if refresh_all:
        LOGGER.info("Cache policy %s: recomputing prediction artifacts", cache_policy)
        pipeline.prepare_baseline(
            config,
            force=True,
        )
        pipeline.train_semantic_multitask(
            config,
            force=True,
        )
        pipeline.train_semantic_bge(
            config,
            force=True,
        )
        pipeline.train_lexical_tfidf(
            config,
            force=True,
        )
        return

    validators = [
        (
            "dual-source baseline",
            validate_baseline_cache,
            lambda: pipeline.prepare_baseline(
                config,
                force=True,
            ),
        ),
        (
            "semantic multitask",
            validate_semantic_multitask_cache,
            lambda: pipeline.train_semantic_multitask(
                config,
                force=True,
            ),
        ),
        (
            "semantic BGE",
            validate_semantic_bge_cache,
            lambda: pipeline.train_semantic_bge(
                config,
                force=True,
            ),
        ),
        (
            "lexical TF-IDF",
            validate_lexical_tfidf_cache,
            lambda: pipeline.train_lexical_tfidf(
                config,
                force=True,
            ),
        ),
    ]
    for name, validator, refresh in validators:
        try:
            validator(config)
            LOGGER.info("Valid cache: %s", name)
        except CacheValidationError as exc:
            LOGGER.warning("%s cache is not usable: %s", name, exc)
            LOGGER.warning("Recomputing %s artifacts", name)
            refresh()
            validator(config)


def predict_submission(
    config: dict[str, Any],
    *,
    output: Path | None = None,
    cache_policy: str | None = None,
) -> Path:
    """Build a submission, recomputing probability artifacts when needed."""
    policy = cache_policy or config["runtime"].get("cache_policy", "auto")
    ensure_prediction_artifacts(config, policy)
    return pipeline.build_submission(config, output)
