"""Gold tier: curated hourly NL analytical dataset (see :mod:`dataset`)."""

from .dataset import GoldHourlyRow, GoldInputs, build_gold_hourly, write_gold, read_gold
from .residual_load import ResidualResult, compute_residual, category_hourly

__all__ = [
    "GoldHourlyRow",
    "GoldInputs",
    "build_gold_hourly",
    "write_gold",
    "read_gold",
    "ResidualResult",
    "compute_residual",
    "category_hourly",
]
