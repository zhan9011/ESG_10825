from __future__ import annotations

from dataclasses import dataclass


OUTPUT_COLUMNS = [
    "id",
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

TASKS = [
    "promise_status",
    "evidence_status",
    "evidence_quality",
    "verification_timeline",
]

LABELS = {
    "promise_status": ["No", "Yes"],
    "evidence_status": ["No", "Yes"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
    ],
}

SCORING_LABELS = {
    task: labels if task == "promise_status" else [*labels, "N/A"]
    for task, labels in LABELS.items()
}

TASK_WEIGHTS = {
    "promise_status": 0.20,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
    "verification_timeline": 0.15,
}

ENSEMBLE_WEIGHTS = {
    "promise_status": {
        "mdeberta": 0.25,
        "roberta_base": 0.25,
        "roberta_large": 0.50,
    },
    "evidence_status": {
        "mdeberta": 0.50,
        "roberta_large": 0.50,
    },
    "evidence_quality": {
        "macbert_large": 0.25,
        "roberta_large": 0.25,
        "roberta_base_quality_binary": 0.50,
    },
    "verification_timeline": {
        "roberta_base": 0.50,
        "roberta_large": 0.50,
    },
}


@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    pretrained_model: str
    task: str
    epochs: int
    binary: bool = False


MODEL_SPECS = [
    ModelSpec("mdeberta", "microsoft/mdeberta-v3-base", "promise_status", 5),
    ModelSpec("mdeberta", "microsoft/mdeberta-v3-base", "evidence_status", 6),
    ModelSpec("roberta_base", "hfl/chinese-roberta-wwm-ext", "promise_status", 4),
    ModelSpec(
        "roberta_base",
        "hfl/chinese-roberta-wwm-ext",
        "verification_timeline",
        4,
    ),
    ModelSpec(
        "roberta_base_quality_binary",
        "hfl/chinese-roberta-wwm-ext",
        "evidence_quality",
        8,
        binary=True,
    ),
    ModelSpec(
        "macbert_large",
        "hfl/chinese-macbert-large",
        "evidence_quality",
        7,
    ),
    ModelSpec(
        "roberta_large",
        "hfl/chinese-roberta-wwm-ext-large",
        "promise_status",
        4,
    ),
    ModelSpec(
        "roberta_large",
        "hfl/chinese-roberta-wwm-ext-large",
        "evidence_status",
        5,
    ),
    ModelSpec(
        "roberta_large",
        "hfl/chinese-roberta-wwm-ext-large",
        "evidence_quality",
        6,
    ),
    ModelSpec(
        "roberta_large",
        "hfl/chinese-roberta-wwm-ext-large",
        "verification_timeline",
        6,
    ),
]
