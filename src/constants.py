"""
src/constants.py - Consolidated constants and dynamic header generation for n-gram models.
This module provides essential constants and a function to dynamically generate the CSV header
based on current model configuration dictionaries.

"""

from model_configs import get_model_configs


def get_header():
    """
    Dynamically generates the header for output CSV based on model configuration dictionaries.

    The header consists of the fixed columns ['word', 'word_len'] followed by the names
    from each model configuration.

    Returns:
        list: The dynamic header list.
    """
    base_header = ["word", "word_len"]
    configs = get_model_configs()
    model_names = [config["name"] for config in configs]
    return base_header + model_names


MAX_WORD_LEN = 100

# End of src/constants.py
