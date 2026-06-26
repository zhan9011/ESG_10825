from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from esg.config import (
    LABELS,
    OUTPUT_COLUMNS,
    SCORING_LABELS,
    TASKS,
    TASK_WEIGHTS,
)


TASK_PROMPTS = {
    "promise_status": "任務：判斷內容是否表達企業對行動的承諾，而非單純描述事實、成果、背景或產品。",
    "evidence_status": "任務：判斷承諾是否附有具體行動、制度、方法、數據、措施或執行紀錄作為支持證據。",
    "evidence_quality": "任務：判斷支持證據是否具體清晰可驗證；籠統方向或模糊行動屬於不清晰。",
    "verification_timeline": "任務：以永續報告發布年 2024 年為基準，判斷承諾適合驗證的時間範圍。",
}

TIME_TERMS = [
    "已",
    "目前",
    "現行",
    "持續",
    "每年",
    "定期",
    "短期",
    "中期",
    "長期",
    "未來",
    "預計",
    "目標",
    "淨零",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_data(path: Path, labeled: bool) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"id": str, "ticker": str}).fillna("N/A")
    required = {"id", "data", "company"}
    if labeled:
        required.update(TASKS)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if labeled:
        data["verification_timeline"] = data["verification_timeline"].replace(
            {"longer_than_5_years": "more_than_5_years"}
        )
    return data


def combined_labeled(train_path: Path, validation_path: Path) -> pd.DataFrame:
    return pd.concat(
        [read_data(train_path, True), read_data(validation_path, True)],
        ignore_index=True,
    )


def applicable_mask(data: pd.DataFrame, task: str) -> pd.Series:
    if task == "promise_status":
        return pd.Series(True, index=data.index)
    if task in {"evidence_status", "verification_timeline"}:
        return data["promise_status"] == "Yes"
    return data["evidence_status"] == "Yes"


def time_features(text: str) -> str:
    years = sorted(set(re.findall(r"20\d{2}", text)))
    terms = [term for term in TIME_TERMS if term in text]
    return f"年份：{','.join(years) or '無'}。時間詞：{','.join(terms) or '無'}。"


def task_text(data: pd.DataFrame, task: str) -> list[str]:
    texts = []
    for row in data.itertuples(index=False):
        prefix = TASK_PROMPTS[task]
        if task == "verification_timeline":
            prefix += time_features(row.data)
        texts.append(f"{prefix} 公司：{row.company}。內容：{row.data}")
    return texts


def enforce_logic(prediction: pd.DataFrame) -> pd.DataFrame:
    prediction = prediction.copy()
    no_promise = prediction["promise_status"] == "No"
    prediction.loc[
        no_promise,
        ["verification_timeline", "evidence_status", "evidence_quality"],
    ] = "N/A"

    has_promise = ~no_promise
    prediction.loc[
        has_promise & (prediction["verification_timeline"] == "N/A"),
        "verification_timeline",
    ] = "already"
    prediction.loc[
        has_promise & (prediction["evidence_status"] == "N/A"),
        "evidence_status",
    ] = "Yes"

    no_evidence = prediction["evidence_status"] != "Yes"
    prediction.loc[no_evidence, "evidence_quality"] = "N/A"
    prediction.loc[
        ~no_evidence & (prediction["evidence_quality"] == "N/A"),
        "evidence_quality",
    ] = "Clear"
    return prediction


def probabilities_to_prediction(
    probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    prediction = pd.DataFrame()
    for task in TASKS:
        prediction[task] = [
            LABELS[task][index] for index in probabilities[task].argmax(axis=1)
        ]
    return enforce_logic(prediction)


def official_metrics(y_true: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    metrics = {}
    weighted_score = 0.0
    for task in TASKS:
        macro_f1 = f1_score(
            y_true[task],
            prediction[task],
            labels=SCORING_LABELS[task],
            average="macro",
            zero_division=0,
        )
        metrics[task] = {
            "macro_f1": float(macro_f1),
            "accuracy": float(accuracy_score(y_true[task], prediction[task])),
            "samples": int(len(y_true)),
        }
        weighted_score += TASK_WEIGHTS[task] * macro_f1
    metrics["weighted_macro_f1"] = float(weighted_score)
    return metrics


def validate_submission(
    submission: pd.DataFrame,
    expected_ids: list[str] | None = None,
) -> None:
    if submission.columns.tolist() != OUTPUT_COLUMNS:
        raise ValueError(f"Submission columns must be {OUTPUT_COLUMNS}")
    if submission.isna().any().any():
        raise ValueError("Submission contains NaN")
    if expected_ids is not None and submission.id.astype(str).tolist() != expected_ids:
        raise ValueError("Submission IDs do not match test IDs")

    allowed = {
        "promise_status": {"Yes", "No"},
        "verification_timeline": set(LABELS["verification_timeline"]) | {"N/A"},
        "evidence_status": {"Yes", "No", "N/A"},
        "evidence_quality": set(LABELS["evidence_quality"]) | {"N/A"},
    }
    for column, values in allowed.items():
        invalid = set(submission[column].astype(str)) - values
        if invalid:
            raise ValueError(f"{column} contains invalid values: {sorted(invalid)}")

    no_promise = submission["promise_status"] == "No"
    dependent = ["verification_timeline", "evidence_status", "evidence_quality"]
    if not (submission.loc[no_promise, dependent] == "N/A").all().all():
        raise ValueError("Rows with promise_status=No must use N/A downstream")
    if (
        submission.loc[~no_promise, ["verification_timeline", "evidence_status"]]
        == "N/A"
    ).any().any():
        raise ValueError("Promise rows require timeline and evidence predictions")
    no_evidence = submission["evidence_status"] != "Yes"
    if not (submission.loc[no_evidence, "evidence_quality"] == "N/A").all():
        raise ValueError("Rows without evidence must use evidence_quality=N/A")
    if (submission.loc[~no_evidence, "evidence_quality"] == "N/A").any():
        raise ValueError("Rows with evidence require an evidence_quality prediction")


def save_submission(
    test: pd.DataFrame,
    prediction: pd.DataFrame,
    output: Path,
) -> None:
    result = pd.concat(
        [test.id.astype(str).reset_index(drop=True), prediction.reset_index(drop=True)],
        axis=1,
    )[OUTPUT_COLUMNS]
    validate_submission(result, test.id.astype(str).tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8")
    print(f"Saved {output} with {len(result)} rows")


def save_metrics(metrics: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
