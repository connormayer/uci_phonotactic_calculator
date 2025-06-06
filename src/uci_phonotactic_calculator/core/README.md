# core

*Purpose* – This package contains the central data structures, configuration management, corpus handling, and variant generation logic for the phonotactic calculator.

See the [parent README](../../README.md) for overall source context.

## What’s inside
- **`config.py`** – Defines the main configuration dataclass (`Config`) that holds all model and processing parameters.
- **`corpus.py`** – Handles loading, parsing, and representing phonological corpora (wordlists).
- **`variants.py`** – Generates all unique combinations of models and configurations (`Variant` objects) to be run, based on the corpus and config.
- **`registries.py`** – Manages registries for pluggable components like models, counting/position strategies, and CLI extensions, allowing for dynamic discovery.
- **`header_schema.py`** & **`header_utils.py`** – Define and manage the schema for output CSV headers, ensuring uniqueness and consistency across different variants.
- **`probability.py`** – Provides utility functions for probability calculations, often used in scoring logic.
- **`filter_aliases.py`** – Manages aliases for configuration filter keys, used by the CLI to select specific variants.
- **`__init__.py`** – Makes the `core` directory a Python package.

These components inter-operate: `corpus.py` loads the input data, `config.py` provides the parameters for processing, and `variants.py` combines these to generate specific calculation tasks (`Variant` instances). The `registries.py` then helps in dispatching these tasks to the appropriate plugin implementations.

## When you’d edit this folder
You would edit files here when changing fundamental data structures (like how configurations or corpora are represented), modifying how model variants are generated, or adjusting the core processing logic that orchestrates model execution.
