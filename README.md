# UCI Phonotactic Calculator

Easily score wordlists with classic and positional-n-gram phonotactic models — right from the command line or a friendly web interface.

---

## ✨ What can I do with it?

| Task | One-liner |
|------|-----------|
| Score a test list with the default model | `python -m uci_phonotactic_calculator.main train.csv test.csv out.csv` |
| Try the demo data set, using an english training file and english test file | `make demo` |
| Launch a Django web interface | `make django` |
| Launch an interactive web UI (Gradio) | `make web` |

The output is a CSV that adds phonotactic scores next to each word, ready for Excel or Pandas.

---

## 🚀 Quick install

```bash
# 1. (Optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 2. Install the core package
pip install uci-phonotactic-calculator
```

That’s it!
If you need the web UI, just add the extra tag:

```bash
pip install "uci-phonotactic-calculator[ui]"
```

---

### 🛠️ Using the `make` shortcuts

The repository ships with a `Makefile` that bundles handy commands such as `make demo`, `make web`, and `make django`.  
To use these targets you will need **Poetry** and **GNU Make** installed on your system.

| Tool | macOS | Windows |
|------|-------|---------|
| Poetry | `curl -sSL https://install.python-poetry.org | python3 -` | Install via the [official installer](https://python-poetry.org/) or simply `pip install poetry` |
| GNU Make | Comes with Xcode CLT (`xcode-select --install`) or `brew install make` | `choco install make` (Chocolatey) or use Git Bash/WSL which already ships with `make` |

If you prefer not to install **Make** you can run the underlying commands directly:

```bash
# Launch the Gradio UI
poetry run python -m uci_phonotactic_calculator.web.web_demo  # same as: make web

# Launch the Django web interface
poetry run python -m uci_phonotactic_calculator.web.django.webcalc  # same as: make django
```

---

## 🏃 Your first run

```bash
# Train on English, score the sample test set, write results to scores.csv
python -m uci_phonotactic_calculator.main data/english.csv \
       data/sample_test_data/english_test_data.csv \
       scores.csv
```

Don’t have your own data yet? Use the built-in demo corpus:

```bash
python -m uci_phonotactic_calculator.main --use-demo-data scores.csv
```

You’ll get a CSV like:

| word | word\_len | ngram\_bound\_conditional | … |
| ---- | --------- | ------------------------- | - |
| CAT  | 3         | -3.87                     | … |

---

## 🖥️ Django interface (optional)

Prefer point-and-click? Fire up the Django web interface:

```bash
make django      # or: python -m uci_phonotactic_calculator.web.django.webcalc
```

A browser window opens where you can drop CSVs, tweak a few options, and download scores.

---


## 🖥️ Gradio interface (optional)

Prefer point-and-click? Fire up the Gradio UI:

```bash
make web      # or: python -m uci_phonotactic_calculator.web.web_demo
```

A browser window opens where you can drop CSVs, tweak a few options, and download scores.

---

## 📚 Want to go deeper?

* Run `python -m uci_phonotactic_calculator.main --help` for **all** flags.
* Developers can install extras with `pip install ".[dev]"` and check out `CONTRIBUTING.md`.
* Full docs & citation info: [https://phonotactics.socsci.uci.edu/](https://phonotactics.socsci.uci.edu/)

---

## ✏️ Citation

If this tool helps your research, please cite:

> Mayer, C., Kondur, A., & Sundara, M. (2022). *UCI Phonotactic Calculator* (v0.1.0). [https://doi.org/10.5281/zenodo.7443706](https://doi.org/10.5281/zenodo.7443706)
