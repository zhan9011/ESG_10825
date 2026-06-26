from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure project logging once for command-line entry points."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
