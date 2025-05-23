# UCI Phonotactic Calculator

[![DOI](https://zenodo.org/badge/476894916.svg)](https://zenodo.org/badge/latestdoi/476894916)

This repository contains the source code for the [UCI Phonotactic Calculator website](https://phonotactics.socsci.uci.edu/), as well as a flexible, extensible CLI for phonotactic modeling and scoring.

- Source code: [`src/`](./src)
- Example datasets: [`data/`](./data)

---

## 🚀 Quick Start

### Requirements
- Python 3.8+
- (Optional) Create a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  ```
- Install dependencies:
  ```bash
  # Install basic dependencies
  pip install -r requirements.txt
  
  # Or install the package with development dependencies
  pip install -e ".[dev,ui]"
  
  # To include Django web interface
  pip install -e ".[dev,ui,web]"
  ```

### Running a Model

```bash
python -m uci_phonotactic_calculator.main <train_file> <test_file> <output_file> [OPTIONS]
```
Example:
```bash
python -m uci_phonotactic_calculator.main data/english.csv data/sample_test_data/english_test_data.csv output.csv --boundary-mode both --aggregate sum --weight-mode legacy_log
```

### Running the Web Interface

#### Gradio UI
```bash
# Using the make command
make web

# Or directly
python -m uci_phonotactic_calculator.web.web_demo
```

#### Django Web Interface
```bash
# Using the make command
make django-web

# Or using the entry point
uci-phonotactic-web runserver

# Or directly
python -m uci_phonotactic_calculator.web.django.manage runserver
```

---

## 🛠️ CLI Options

The CLI supports a wide range of options. Run `python -m src.main --help` for a full list.

### 🔎 Filtering and Discovering Keys

You can restrict the grid search to specific model/configuration variants using one or more `--filter` flags:

```sh
python -m src.main ... --filter KEY=VAL [--filter KEY2=VAL2 ...]
```
- Each filter restricts the grid to configs where `Config.<KEY> == <VAL>`.
- Repeat the flag to combine filters (logical AND).
- Examples:
  ```sh
  --filter smoothing=laplace --filter n=2
  --filter aggregate=logsumexp
  ```
- Both long and short key aliases are accepted (e.g., `n` for `ngram_order`, `prob` for `prob_mode`).
- For a full list of accepted keys and aliases, run:
  ```sh
  python -m src.main --list-filters
  ```
  This will print all canonical keys and their available aliases.

---

### 🐍 Scripting & CI: Disabling Progress Bars

To make scripting and automation easier (e.g., in Docker or CI), you can suppress all progress bars globally by setting the environment variable:

```sh
NO_PROGRESS=1
```

This disables all Rich progress bars, regardless of CLI flags. You can also use the `--no-progress` CLI flag for one-off runs.

**Key flags:**
- `--boundary-mode`: `both` (default), `prefix`, `suffix`, `none`
- `--aggregate`: `sum`, `mean`, `min`, `max`, `none`
- `--weight-mode`: `none`, `raw`, `log`, `legacy_log`
- `--position-strategy`: `absolute`, `relative`, `none` (default is None)
- `--smoothing-scheme`: `laplace`, `none`, `kn`
- `--count-strategy`: `ngram` (default), others as available
- `--prob-mode`: `conditional`, `joint`
- `--filter KEY=VAL`: Filter variants by config
- `--no-color`: Disable colored CLI output

Example:
```bash
python -m src.main data/english.csv data/sample_test_data/english_test_data.csv output.csv \
    --boundary-mode both --aggregate sum --weight-mode legacy_log --position-strategy absolute
```

---

## 📋 Output CSV & Header Logic

- **Headers:** Output CSV headers are now always unique and schema-driven. All configuration axes are included, and legacy aliasing is removed for clarity.
- **No duplicate headers:** The system guarantees that every configuration generates a unique header.
- **Debugging:** Set the environment variable `DEBUG_VARIANTS=1` to log all generated headers and configurations for troubleshooting.

---

## ⚖️ Weighting & Smoothing

- **Weighting modes:**
    - `none`: Unweighted (1.0)
    - `raw`: Raw frequency
    - `log`: log(freq + 1)
    - `legacy_log`: 2018 behavior (log(freq) if freq > 0, -inf if freq == 0)
- **Smoothing:**
    - Laplace smoothing (`laplace`) now automatically zeroes negative or -inf counts (from legacy_log) before smoothing, ensuring legacy compatibility.

---

## 🧑‍💻 Extending the CLI

You can extend the CLI by registering your own argument injectors using the `@register_cli_ext` decorator from `src.cli_ext`. This allows you to add custom flags or argument groups from external packages or add-ons.

### Example: Adding a Custom CLI Extension

Suppose you want to add a new CLI flag `--my-flag` that prints a custom message. You can do this in an external Python file or package:

```python
# my_cli_plugin.py
from src.cli_ext import register_cli_ext
import argparse

@register_cli_ext("myplugin")
class MyPlugin:
    def flags(self):
        action = argparse.Action(
            option_strings=["--my-flag"],
            dest="my_flag",
            nargs=0,
            help="Print a custom message from my plugin."
        )
        return [action]
```

When you run the main CLI, this flag will appear automatically if your plugin is imported before CLI construction.

---

## 🧩 Writing a Custom Model or Strategy

The system is fully registry-driven. You can add new models, weighting, smoothing, or aggregation strategies without changing core code.

### Example: Registering a Custom Aggregator

```python
# my_aggregators.py
from src.plugins.core import register

def my_custom_agg(scores):
    # Your aggregation logic here
    return sum(scores) / (len(scores) + 1)

register('aggregate_mode', 'my_custom_agg')(my_custom_agg)
```

Now you can use `--aggregate my_custom_agg` in the CLI.

### Example: Registering a Custom Model

```python
# my_model.py
from src.plugins.core import register, BaseModel

@register('model', 'my_ngram_model')
class MyNGramModel(BaseModel):
    def fit(self, corpus):
        # Custom training logic
        ...
    def score(self, token):
        # Custom scoring logic
        ...
```

Your model will now be available as a `--model my_ngram_model` option.

---

## 🐞 Error Handling & UX

- **Interrupts:** Pressing Ctrl+C exits gracefully with a yellow, bold `[Interrupted by user]` message.
- **Warnings:** Deprecated use of the string `'none'` for `position_strategy` is now eliminated; use Python `None` or omit the flag.
- **Helpful errors:** Permission and file errors are reported clearly.

---

## 🧰 Testing & Debugging

- Use `DEBUG_VARIANTS=1` to log all headers/configs for debugging duplicate header issues.
- All errors and warnings are designed to be clear and actionable.

---

## 📝 Development Workflow

This project uses a Makefile to streamline development tasks. Here are the available commands:

```bash
# Run linting and tests
make

# Clean build artifacts
make clean

# Run linting tools (ruff, black, mypy)
make lint

# Format code automatically
make format

# Run tests
make test

# Build distribution package
make build

# Run the web UI
make web

# Run the calculator with demo data
make demo

# Show all available commands
make help
```

### Development Setup

For development, install the package with all optional dependencies:

```bash
pip install -e ".[dev,ui]"
```

This will install all the necessary tools for development, including:

- **black**: Code formatter
- **ruff**: Fast Python linter
- **mypy**: Static type checker
- **pytest**: Testing framework
- **build**: Package building tool
- **gradio**: UI framework for the web interface

### Web Interface

To run the web interface:

```bash
# Using make
make web

# Or using Python module directly
python -m uci_phonotactic_calculator.web
```

---

## 📖 Citing the UCI Phonotactic Calculator

If you publish work that uses the UCI Phonotactic Calculator, please cite this repository:

> Mayer, C., Kondur, A., & Sundara, M. (2022). UCI Phonotactic Calculator (Version 0.1.0) [Computer software]. https://doi.org/10.5281/zenodo.7443706

---

## 🤝 Contributing

Contributions and suggestions are welcome! Please open issues or pull requests for bugfixes, improvements, or new features.

---

## 🔗 Resources
- [UCI Phonotactic Calculator Website](https://phonotactics.socsci.uci.edu/)
- [Project on Zenodo](https://doi.org/10.5281/zenodo.7443706)

```python
from src.cli_ext import register_cli_ext, CLIExtension

@register_cli_ext('my_plugin')
class MyCLIExt:
    def inject(self, parser):
        parser.add_argument('--my-flag', action='store_true', help='Enable my feature')
```

This will automatically inject your flag into the CLI when your extension is imported.

## Adding a new aggregator

When implementing a custom aggregator or counter, ensure your class uses the ABC-friendly accumulate signature:

```python
from src.plugins.strategies.base import BaseCounter

class MyCounter(BaseCounter):
    def accumulate(self, token, weight, **kwargs):
        # Your accumulation logic here
        pass
```

The `**kwargs` ensures compatibility with the abstract base class and the CLI, allowing extra arguments like `boundary` to be passed without error.

For more details, see the source code or run `python -m src.ngram_calculator --help` to view extensions and available flags.

---

All CLI examples and documentation now use the new flag names and header tokens for consistency and easier integration with tools like Pandas.
