from __future__ import annotations

import pandas as pd

from esg.data import validate_submission


def validate_output(submission: pd.DataFrame, expected_ids: list[str]) -> None:
    """Validate final submission format and label consistency."""
    validate_submission(submission, expected_ids)
