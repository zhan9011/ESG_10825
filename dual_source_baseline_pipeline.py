from __future__ import annotations

from esg.pipeline import pipeline_parser, run_pipeline


DUAL_SOURCE_FULL_WEIGHT = 0.80


if __name__ == "__main__":
    args = pipeline_parser(
        name="Dual-source 80/20 baseline",
        default_output="outputs/dual_source_baseline_retrained.csv",
    ).parse_args()
    run_pipeline(args, DUAL_SOURCE_FULL_WEIGHT)
