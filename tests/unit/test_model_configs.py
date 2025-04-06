# tests/unit/test_model_configs.py
"""
test_model_configs.py - Unit tests for model configuration dictionaries.
This module verifies that get_model_configs returns properly structured configurations
with all required keys and correct naming conventions.
"""

import pytest
from src.model_configs import get_model_configs

def test_get_model_configs_structure():
    configs = get_model_configs()
    assert isinstance(configs, list) and len(configs) > 0
    required_keys = {"name", "model", "position", "smoothed", "token_weighted", "aggregation"}
    for config in configs:
        assert required_keys.issubset(set(config.keys()))
        if config["model"] == "bigram":
            assert "use_boundaries" in config
            assert "conditional" in config
        assert "aggregation" in config

@pytest.mark.parametrize("config", get_model_configs())
def test_config_naming_conventions(config):
    name = config["name"]
    if config.get("smoothed", False):
        assert "smoothed" in name
    if config.get("token_weighted", False):
        assert "freq_weighted" in name
    