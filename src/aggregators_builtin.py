# src/aggregators_builtin.py
from .aggregate import (
    log_product, logsumexp,
    linear_sum, linear_sum_plus1,
    min_val, max_val,
)
from .registries import register

register('aggregate_mode', 'prod')(log_product)
register('aggregate_mode', 'lse')(logsumexp)
register('aggregate_mode', 'sum')(linear_sum)
register('aggregate_mode', 'sum_plus1')(linear_sum_plus1)
# Alias removed; callers must use 'sum_plus1'.
register('aggregate_mode', 'min')(min_val)
register('aggregate_mode', 'max')(max_val)
