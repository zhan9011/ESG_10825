from __future__ import annotations

from pathlib import Path

import pandas as pd

from esg.data import combined_labeled, read_data


def load_labeled(train_path: Path, validation_path: Path) -> pd.DataFrame:
    """Load the combined train and validation data."""
    return combined_labeled(train_path, validation_path)


def load_unlabeled(test_path: Path) -> pd.DataFrame:
    """Load unlabeled inference data."""
    return read_data(test_path, labeled=False)
