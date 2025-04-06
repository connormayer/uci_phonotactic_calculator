"""
src/model_configs.py - Module to generate model configuration dictionaries for n-gram models.
Provides a unified configuration for unigram and bigram models with explicit conditions:
Positional vs. Non-Positional, Word Boundaries (wb) vs. No Boundaries (nwb),
and Conditional vs. Joint (with aggregation for positional models).

"""

from itertools import product

def get_model_configs():
    """
    Generates a list of configuration dictionaries for each n-gram model variant.
    The configurations cover:
      - Unigram non-positional (log and joint)
      - Unigram positional (always joint, with aggregation: sum or prod)
      - Bigram conditional and joint in both positional and non-positional forms,
        with options for word boundaries.
    Returns:
      A list of dictionaries, each representing a model configuration.
    """
    configs = []
    
    # Unigram non-positional: sum aggregation
    name = "uni_nonpos_sum"
    for smoothed, token_weighted in product([False], repeat=2):
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        configs.append({
            "name": name,
            "model": "unigram",
            "position": "non_positional",
            "smoothed": smoothed,
            "token_weighted": token_weighted,
            "aggregation": "sum"
        })
    
    # Unigram non-positional: mult aggregation
    name = "uni_nonpos_prod"
    for smoothed, token_weighted in product([False], repeat=2):
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        configs.append({
            "name": name,
            "model": "unigram",
            "position": "non_positional",
            "smoothed": smoothed,
            "token_weighted": token_weighted,
            "aggregation": "prod"
        })

    # Unigram positional: sum aggregation
    name = "uni_pos_sum"
    for smoothed, token_weighted in product([False], repeat=2):
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        configs.append({
            "name": name,
            "model": "unigram",
            "position": "positional",
            "smoothed": smoothed,
            "token_weighted": token_weighted,
            "aggregation": "sum"
        })
    
    # Unigram non-positional: mult aggregation
    name = "uni_pos_prod"
    for smoothed, token_weighted in product([False], repeat=2):
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        configs.append({
            "name": name,
            "model": "unigram",
            "position": "positional",
            "smoothed": smoothed,
            "token_weighted": token_weighted,
            "aggregation": "prod"
        })
    
    # Bigram models:
    # Bigram conditional, positional
    for use_boundaries, smoothed, token_weighted, aggregation in product([True, False], [False], [False], ["sum", "prod"]):
        name = "bi_pos_cond_" + ("wb" if use_boundaries else "nwb") + "_" + aggregation
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        configs.append({
            "name": name,
            "model": "bigram",
            "position": "positional",
            "conditional": True,
            "use_boundaries": use_boundaries,
            "smoothed": smoothed,
            "token_weighted": token_weighted,
            "aggregation": aggregation
        })
        
    # Bigram conditional, non-positional
    for use_boundaries, smoothed, token_weighted, aggregation in product([True, False], [False], [False], ["sum", "prod"]):
        name = "bi_nonpos_cond_" + ("wb" if use_boundaries else "nwb") + "_" + aggregation
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        configs.append({
            "name": name,
            "model": "bigram",
            "position": "non_positional",
            "conditional": True,
            "use_boundaries": use_boundaries,
            "smoothed": smoothed,
            "token_weighted": token_weighted,
            "aggregation": aggregation
        })
    
    # Bigram joint, positional
    for use_boundaries, smoothed, token_weighted, aggregation in product([True, False], [False], [False], ["sum", "prod"]):
        name = "bi_pos_joint_" + ("wb" if use_boundaries else "nwb") + "_" + aggregation
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        configs.append({
            "name": name,
            "model": "bigram",
            "position": "positional",
            "conditional": False,
            "use_boundaries": use_boundaries,
            "smoothed": smoothed,
            "token_weighted": token_weighted,
            "aggregation": aggregation
        })
    
    # Bigram joint, non-positional
    for use_boundaries, smoothed, token_weighted, aggregation in product([True, False], [False], [False], ["sum", "prod"]):
        name = "bi_nonpos_joint_" + ("wb" if use_boundaries else "nwb") + "_" + aggregation
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        configs.append({
            "name": name,
            "model": "bigram",
            "position": "non_positional",
            "conditional": False,
            "use_boundaries": use_boundaries,
            "smoothed": smoothed,
            "token_weighted": token_weighted,
            "aggregation": aggregation
        })
    
    return configs
