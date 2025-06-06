# plugins

*Purpose* – This package provides the infrastructure for extending the calculator's functionality with new models, strategies, and transformations, and includes several built-in implementations.

See the [parent README](../../README.md) for overall source context.

## What’s inside
- **`core.py`** – Defines base classes for plugins (e.g., `BaseModel`, `BaseStrategy`) and the registration mechanism (e.g., `@register` decorator) used to make custom components discoverable by the main application.
- **`ngram_model.py`** – A built-in phonotactic model based on n-gram statistics.
- **`neighbourhood.py`** – A built-in model/strategy related to phonological neighborhood density calculations.
- **`prob_transforms/`** – A sub-package containing various plugins for transforming probabilities or scores (e.g., log transform, Z-score). See its [README.md](./prob_transforms/README.md) for details.
- **[`strategies/`](./strategies/README.md)** – A sub-package for different counting and positional strategies. See its [README.md](./strategies/README.md).
- **`mixins.py`** – Contains reusable helper classes (mixins) for plugin development.
- **`fallback.py`** – Provides default or fallback plugin implementations.
- **`utils/`** – Utility functions specific to plugin operations.
- **`__init__.py`** – Makes the `plugins` directory a Python package and often imports key plugins to ensure they are registered.

## How to write your own plugin (in three steps)
1.  **Define your class**: Create a Python class that inherits from an appropriate base class in `plugins.core` (e.g., `BaseModel` for a new phonotactic model, or a specific strategy base from `plugins.strategies.base`).
2.  **Implement required methods**: Override the necessary methods from the base class to implement your custom logic (e.g., `fit()` and `score()` for a model).
3.  **Register your plugin**: Use the `@register('plugin_type', 'your_plugin_name')` decorator (from `plugins.core`) on your class to make it available to the calculator. Ensure the module containing your plugin is imported somewhere (e.g., in `plugins/__init__.py` or your project's main entry point) so the registration code runs.

## When you’d edit this folder
You would add new files or sub-packages here when implementing new phonotactic models, scoring strategies, or other extensible components. You might edit `core.py` if changing the fundamental plugin registration system.
