from __future__ import annotations

import pandas as pd


def company_text(data: pd.DataFrame) -> list[str]:
    """Build the common company/text representation for non-transformer features."""
    return [f"{row.company} [SEP] {row.data}" for row in data.itertuples(index=False)]
