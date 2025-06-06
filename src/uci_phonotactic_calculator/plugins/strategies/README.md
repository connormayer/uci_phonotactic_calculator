# strategies

*Purpose* – This package defines and implements various strategies for counting phonotactic elements (e.g., n-grams) and considering their positions within a word.

See the [parent README](../README.md) for plugin system context and the [grandparent README](../../../README.md) for overall source context.

## What’s inside
- **`base.py`** – Defines the base classes for different types of strategies (e.g., `BaseCountStrategy`, `BasePositionStrategy`). New strategies should inherit from these.
- **`ngram.py`** – Implements strategies for counting n-grams (e.g., simple n-gram counts, weighted counts). This is a type of **count strategy**.
- **`position.py`** – Implements strategies for how positional information is incorporated into phonotactic scores (e.g., absolute position, relative position, no positional information). This is a type of **position strategy**.
- **`__init__.py`** – Makes the `strategies` directory a Python package and typically imports the implemented strategies to ensure they are registered with the plugin system.

## Count Strategies vs. Position Strategies

-   **Count Strategies** (e.g., in `ngram.py`): Determine *what* phonotactic elements are counted and *how* their raw frequencies or weights are determined. For example, a count strategy might define how to extract bigrams from a word and whether to use their raw frequency or a log-transformed frequency.
-   **Position Strategies** (e.g., in `position.py`): Determine *how the location* of a phonotactic element within a word influences its contribution to the overall score. For example, a position strategy might assign different weights to n-grams based on whether they appear at the beginning, middle, or end of a word, or it might ignore position altogether.

These two types of strategies often work together. A phonotactic model might use an n-gram count strategy to get the basic frequencies of n-grams and then apply a position strategy to modify those scores based on where the n-grams occur.

## How to register new strategies
1.  Create a new Python module in this directory (or a sub-directory).
2.  Define your strategy class, inheriting from an appropriate base class in `base.py`.
3.  Implement the required methods.
4.  Register your strategy using the `@register('count_strategy', 'your_name')` or `@register('position_strategy', 'your_name')` decorator from `uci_phonotactic_calculator.plugins.core`.
5.  Ensure your new module is imported in `strategies/__init__.py` or another appropriate `__init__.py` file.

## When you’d edit this folder
You would add or modify files here when implementing new ways to count phonotactic units (like different types of n-grams or features) or new methods for incorporating positional information into scores.
