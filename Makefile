.PHONY: clean lint test build docs web django-web demo all-variants legacy help

# Default action when running `make` without arguments
all: lint test

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	python -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('dist') + glob.glob('build') + glob.glob('*.egg-info') + glob.glob('.coverage') + glob.glob('htmlcov') + glob.glob('.pytest_cache') + glob.glob('.ruff_cache') + glob.glob('.mypy_cache') + glob.glob('**/__pycache__', recursive=True) if os.path.exists(p)]; [os.remove(p) for p in glob.glob('**/*.pyc', recursive=True) if os.path.exists(p)]"
	@echo "Clean completed successfully!"

# Run linting tools
lint:
	ruff check --fix .
	ruff format
	mypy --strict .

# Auto-format code
format:
	ruff format
	ruff check --fix .

# Run tests
test:
	python -m pytest

# Build distributable package
build: clean
	python -m build

# Run the web UI
web:
	python -c "from uci_phonotactic_calculator.web.gradio.web_demo_v2 import main; main()"

# Run the Django web interface
django-web:
	@echo "Starting UCI Phonotactic Calculator Django UI..."
	@python -c "import webbrowser; import time; time.sleep(0.5); webbrowser.open('http://127.0.0.1:8000/')"
	python -m uci_phonotactic_calculator.web.django.manage runserver

# Run the demo calculator
demo:
	python -m uci_phonotactic_calculator.cli.main --use-demo-data output.csv

# Run ngram calculator with all variants (--all flag)
all-variants:
	@echo "Running UCI Phonotactic Calculator with all variants..."
	python -m uci_phonotactic_calculator.cli.main --use-demo-data --all output.csv

# Run the legacy calculator variant (default mode with 16-column output)
legacy:
	@echo "Running UCI Phonotactic Calculator legacy variant..."
	@echo "Note: Legacy mode is now the default (16-column output)."
	python -m uci_phonotactic_calculator.cli.main --use-demo-data output.csv

# Show help
help:
	@echo "Available commands:"
	@echo "  make clean      - Remove build artifacts and cache directories"
	@echo "  make lint       - Run linting checks"
	@echo "  make format     - Auto-format code"
	@echo "  make test       - Run tests"
	@echo "  make build      - Build distribution package"
	@echo "  make web        - Run the Gradio web UI (web_demo)"
	@echo "  make django-web - Run the Django web UI"
	@echo "  make demo       - Run the calculator with demo data"
	@echo "  make all-variants - Run ngram calculator with all variants (--all flag)"
	@echo "  make legacy     - Run the legacy calculator variant"
