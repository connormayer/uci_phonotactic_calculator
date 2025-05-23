# src/aggregators_builtin.py
from ..core.registries import register
from .aggregate import (
    linear_sum,
    linear_sum_plus1,
    log_product,
    logsumexp,
    max_val,
    min_val,
)

register("aggregate_mode", "prod")(log_product)
register("aggregate_mode", "lse")(logsumexp)
register("aggregate_mode", "sum")(linear_sum)
register("aggregate_mode", "sum_plus1")(linear_sum_plus1)
# Alias removed; callers must use 'sum_plus1'.
register("aggregate_mode", "min")(min_val)
register("aggregate_mode", "max")(max_val)
