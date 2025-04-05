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
    
    # Unigram non-positional: log mode
    base_log = "uni_prob"
    for smoothed, token_weighted in product([False, True], repeat=2):
        name = base_log
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        configs.append({
            "name": name,
            "model": "unigram",
            "position": "non_positional",
            "joint": False,
            "smoothed": smoothed,
            "token_weighted": token_weighted
        })
    
    # Unigram non-positional: joint mode
    base_joint_nonpos = "uni_joint_nonpos"
    for smoothed, token_weighted in product([False, True], repeat=2):
        name = base_joint_nonpos
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        configs.append({
            "name": name,
            "model": "unigram",
            "position": "non_positional",
            "joint": True,
            "smoothed": smoothed,
            "token_weighted": token_weighted
        })
    
    # Unigram positional (always joint) with aggregation
    base_joint_pos = "uni_joint_pos"
    for smoothed, token_weighted, aggregation in product([False, True], [False, True], ["sum", "prod"]):
        name = base_joint_pos
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        if aggregation == "prod":
            name += "_prod"
        configs.append({
            "name": name,
            "model": "unigram",
            "position": "positional",
            "joint": True,
            "smoothed": smoothed,
            "token_weighted": token_weighted,
            "aggregation": aggregation
        })
    
    # Bigram models:
    # Bigram conditional, positional
    for use_boundaries, smoothed, token_weighted, aggregation in product([True, False], [False, True], [False, True], ["sum", "prod"]):
        base = "bi_cond_pos_" + ("wb" if use_boundaries else "nwb")
        name = base
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        if aggregation == "prod":
            name += "_prod"
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
    for use_boundaries, smoothed, token_weighted in product([True, False], [False, True], [False, True]):
        base = "bi_cond_nonpos_" + ("wb" if use_boundaries else "nwb")
        name = base
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
            "token_weighted": token_weighted
        })
    
    # Bigram joint, positional
    for use_boundaries, smoothed, token_weighted, aggregation in product([True, False], [False, True], [False, True], ["sum", "prod"]):
        base = "bi_joint_pos_" + ("wb" if use_boundaries else "nwb")
        name = base
        if token_weighted:
            name += "_freq_weighted"
        if smoothed:
            name += "_smoothed"
        if aggregation == "prod":
            name += "_prod"
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
    for use_boundaries, smoothed, token_weighted in product([True, False], [False, True], [False, True]):
        base = "bi_joint_nonpos_" + ("wb" if use_boundaries else "nwb")
        name = base
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
            "token_weighted": token_weighted
        })
    
    return configs

# End of src/model_configs.py