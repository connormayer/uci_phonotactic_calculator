"""
Gradio front-end for the UCI Phonotactic Calculator
---------------------------------------------------
✓ Works on Hugging-Face Spaces
✓ Uses the *installed* Python package – no relative "src" hacks
✓ Returns both a preview DataFrame *and* a downloadable CSV
✓ Tabbed interface for better organization
✓ Extensive documentation and examples
"""

import atexit
import functools
import logging
import os
import os.path
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd

# NEW ▶
CSS = """
/* Main container and layout */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
}

#df-wrapper {
    max-height: 400px;
    overflow-y: auto;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

/* Step-by-step wizard styling */
.step-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px rgba(102, 126, 234, 0.1);
}

.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}

.step-number {
    background: rgba(255, 255, 255, 0.2);
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 14px;
}

.step-title {
    font-size: 18px;
    font-weight: 600;
    margin: 0;
}

.step-description {
    margin: 0;
    opacity: 0.9;
    font-size: 14px;
    line-height: 1.4;
}

/* Card-based design */
.input-card {
    background: white;
    border: 1px solid #e1e5e9;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
}

.input-card:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-color: #667eea;
}

.card-title {
    font-size: 16px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-subtitle {
    font-size: 14px;
    color: #64748b;
    margin: 0 0 16px 0;
    line-height: 1.4;
}

/* Enhanced form controls */
.option-group {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}

.option-label {
    font-weight: 600;
    color: #374151;
    margin-bottom: 8px;
    font-size: 14px;
}

.option-divider {
    position: relative;
    text-align: center;
    margin: 20px 0;
    display: flex;
    align-items: center;
    justify-content: center;
}

.option-divider::before {
    content: "";
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, #e2e8f0 20%, #e2e8f0 80%, transparent);
    transform: translateY(-100%);
}

.option-divider span {
    background: white;
    padding: 8px 16px;
    color: #64748b;
    font-weight: 600;
    font-size: 16px;
    border: 2px solid #e2e8f0;
    border-radius: 20px;
    position: relative;
    z-index: 1;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    display: inline-block;
}

/* Enhanced buttons */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 14px 28px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 16px;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
}

.primary-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
}

.secondary-button {
    background: white;
    color: #667eea;
    border: 2px solid #667eea;
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.secondary-button:hover {
    background: #667eea;
    color: white;
}

/* Results styling */
.results-container {
    background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    margin-top: 24px;
    box-shadow: 0 4px 6px rgba(46, 204, 113, 0.1);
}

.results-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}

.results-icon {
    background: rgba(255, 255, 255, 0.2);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}

.download-section {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}

/* Enhanced info boxes */
.info-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 4px solid #2196f3;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 16px;
}

.warning-box {
    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    border-left: 4px solid #ff9800;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 16px;
}

.feature-box {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border-left: 4px solid #9c27b0;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 16px;
}

.success-box {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
    border-left: 4px solid #4caf50;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 16px;
}

/* Loading and progress */
.loading-indicator {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

.loading-spinner {
    width: 20px;
    height: 20px;
    border: 2px solid #e2e8f0;
    border-top: 2px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Enhanced tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

table th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: left;
    padding: 12px;
    font-weight: 600;
}

table td {
    border-top: 1px solid #e2e8f0;
    padding: 12px;
}

table tr:nth-child(even) {
    background: #f8fafc;
}

table tr:hover {
    background: #f1f5f9;
}

/* Results table styling for better visibility */
.results-container table {
    background-color: white !important;
}

.results-container table thead th {
    background-color: #f8f9fa !important;
    color: #2c3e50 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #dee2e6 !important;
    padding: 12px 8px !important;
}

.results-container table tbody td {
    color: #495057 !important;
    padding: 8px !important;
    border-bottom: 1px solid #e9ecef !important;
}

.results-container table tbody tr:hover {
    background-color: #f8f9fa !important;
}

/* Typography improvements */
h1, h2, h3, h4 {
    font-weight: 600;
    color: #1e293b;
    line-height: 1.2;
}

h1 { font-size: 2rem; margin-bottom: 0.5rem; }
h2 { font-size: 1.5rem; margin-bottom: 0.5rem; }
h3 { font-size: 1.25rem; margin-bottom: 0.5rem; }
h4 { font-size: 1.1rem; margin-bottom: 0.5rem; }

/* Utility classes */
.text-center { text-align: center; }
.text-muted { color: #64748b; }
.text-small { font-size: 0.875rem; }
.mb-0 { margin-bottom: 0; }
.mb-2 { margin-bottom: 0.5rem; }
.mb-4 { margin-bottom: 1rem; }
.mt-4 { margin-top: 1rem; }

/* Footer */
footer {
    margin-top: 40px;
    text-align: center;
    font-size: 0.875rem;
    color: #64748b;
    padding: 20px;
    border-top: 1px solid #e2e8f0;
}

/* Four-panel layout */
.calculator-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
}

.calculator-container .gradio-container {
    max-width: 1200px;
    margin: 0 auto;
}

.calculator-container .input-card {
    margin-bottom: 20px;
}

.calculator-container .results-container {
    margin-top: 40px;
}

/* Custom dropdown-style-upload */
.dropdown-style-upload {
    padding: 8px 12px;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    font-size: 14px;
    cursor: pointer;
}

.dropdown-style-upload:hover {
    border-color: #667eea;
}

/* Style UploadButton to look like dropdown */
.dropdown-style-upload button {
    width: 100% !important;
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    font-size: 14px !important;
    text-align: left !important;
    color: #374151 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
}

.dropdown-style-upload button:hover {
    border-color: #667eea !important;
    background: #f8fafc !important;
}

.dropdown-style-upload button::after {
    content: "▼" !important;
    font-size: 12px !important;
    color: #9ca3af !important;
    margin-left: auto !important;
}

/* Override Tailwind's justify-center utility added by Gradio */
.dropdown-style-upload button.justify-center {
    justify-content: flex-start !important;
}

/* Upload CSV button alignment */
.upload-csv-button button {
    text-align: left !important;
    justify-content: flex-start !important;
}

.upload-csv-button {
    text-align: left !important;
}
"""
APP_TITLE = "UCI Phonotactic Calculator"
BANNER_MD = (
    "# 🧮 UCI Phonotactic Calculator\n" "Instant phonotactic scores from n-gram models"
)

# ---> public, documented API wrapper around the CLI
from uci_phonotactic_calculator.cli.demo_data import get_demo_paths
from uci_phonotactic_calculator.cli.legacy import run as ngram_run
from uci_phonotactic_calculator.plugins import PluginRegistry

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Progress tracking for the Gradio UI ---
from uci_phonotactic_calculator.utils.progress_base import GradioProgress

# Configure logging to show DEBUG messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

TMP_DIR = Path(tempfile.gettempdir())


# ---------------------------------------------------------------------
# Back-end helper
# ---------------------------------------------------------------------
def score(
    train_csv,  # gr.File or None
    test_csv,  # gr.File or None
    model,  # str
    run_full_grid,  # bool
    ngram_order,  # int
    filter_string,  # str like "weight_mode=raw prob_mode=joint"
    hide_progress,  # bool
    neighbourhood_extra_args=None,  # Optional extra args for neighbourhood mode
):
    """
    Execute the scorer and return (DataFrame, CSV-path) for Gradio.

    Parameters:
        train_csv: Training corpus CSV file or None if using demo data
        test_csv: Test corpus CSV file or None if using demo data
        model: Model plugin name to use
        run_full_grid: Whether to run all model variants
        ngram_order: The n-gram order to use (1-4)
        filter_string: Space-separated key=value pairs for filtering
        hide_progress: Whether to hide the progress indicator

    Returns:
        Tuple of (DataFrame preview, CSV file path)
    """
    # -------------------- resolve input paths -----------------------
    logger.info(f"Starting scoring with model={model}, n={ngram_order}")

    # Handle packaged files (when train_csv is a string path)
    if isinstance(train_csv, str) and not os.path.isabs(train_csv):
        # When a file name from the package data is selected (not an absolute path)
        import importlib.resources as pkg_resources
        import uci_phonotactic_calculator.data as data_pkg

        # Use the modern importlib.resources approach
        train_path = str(pkg_resources.files(data_pkg) / train_csv)
        logger.info(f"Using packaged training file: {train_path}")

        # Handle test file - check if it's also a packaged file
        if isinstance(test_csv, str) and not os.path.isabs(test_csv):
            # Test file is also a packaged file
            test_path = str(pkg_resources.files(data_pkg) / test_csv)
            logger.info(f"Using packaged test file: {test_path}")
        else:
            # For test file, use the uploaded file
            test_path = test_csv.name if not isinstance(test_csv, str) else test_csv
            logger.info(f"Using test file: {test_path}")
    else:
        # Both custom files
        if train_csv is None or test_csv is None:
            raise gr.Error(
                "Upload BOTH training & test CSVs *or* select a default training file."
            )
        # Support both Gradio File objects and plain string paths
        train_path = train_csv if isinstance(train_csv, str) else train_csv.name
        test_path = test_csv if isinstance(test_csv, str) else test_csv.name
        logger.info(f"Using custom files: {train_path}, {test_path}")

    # ------------------------------------------------------------------
    # Packaged files are just paths - no special behavior
    # ------------------------------------------------------------------
    # default training file is just another path – nothing else changes

    out_file = TMP_DIR / f"scores_{uuid.uuid4().hex}.csv"
    atexit.register(functools.partial(out_file.unlink, missing_ok=True))

    # -------------------- translate filters -------------------------
    filters = {}
    # Initialize extra_args to avoid UnboundLocalError
    extra_args = []
    tokens = filter_string.split()
    if tokens and tokens[0] == "--filter":
        tokens = tokens[1:]  # drop the flag if present
    if tokens:
        for tok in tokens:
            if "=" not in tok:
                raise gr.Error(f"Filter '{tok}' must look like key=value")
            k, v = tok.split("=", 1)
            # 1️⃣ probability radio
            if k == "prob_mode" and v in ("conditional", "joint"):
                logger.info(f"Setting probability mode to {v}")
                # For probability modes, use --prob-mode flag directly
                extra_args.extend(["--prob-mode", v])
                continue  # Skip adding to filters since we're handling it via CLI flag

            # 2️⃣ weight-mode radio
            elif k == "weight_mode":
                logger.info(f"Setting weight mode to {v}")
                extra_args.extend(["--weight-mode", v])
                continue

            # 3️⃣ smoothing radio
            elif k in ("smoothing", "smoothing_scheme"):
                logger.info(f"Setting smoothing scheme to {v}")
                extra_args.extend(["--smoothing-scheme", v])
                continue

            # 4️⃣ neighbourhood mode → real CLI flag
            elif k == "neighbourhood_mode" and v:
                logger.info(f"Setting neighbourhood mode to: {v}")
                extra_args.extend(
                    ["--neighbourhood-mode", v]
                )  # Pass as an actual CLI flag
                continue

            # anything else really is a grid-filter
            else:
                filters[k] = v

    logger.info(f"Filters: {filters}, Extra args: {extra_args}")

    # -------------------- invoke library with Gradio progress patch ---------------------------
    from uci_phonotactic_calculator.utils.progress import progress

    _orig_progress = progress  # keep original

    def _gradio_progress(enabled=True):
        return GradioProgress(enabled=enabled and not hide_progress)

    # Patch the progress function
    import uci_phonotactic_calculator.utils.progress as _p

    _p.progress = _gradio_progress
    try:
        # Combine n-gram order with any other extra args we've set up
        combined_extra_args = ["-n", str(ngram_order)] + extra_args

        # Add neighbourhood mode extra args if provided
        if neighbourhood_extra_args:
            combined_extra_args.extend(neighbourhood_extra_args)

        logger.info(f"Running with extra_args: {combined_extra_args}")

        ngram_run(
            train_file=train_path,
            test_file=test_path,
            output_file=str(out_file),
            model=None if run_full_grid else model,
            run_all=run_full_grid,
            filters=filters,
            show_progress=not hide_progress,  # still disables library chatter
            extra_args=combined_extra_args,
        )
    finally:
        _p.progress = _orig_progress  # guarantee cleanup

    df = pd.read_csv(out_file)
    df_preview = df.head(50).iloc[:, :30]  # show only first 50 rows, 30 cols in UI
    logger.info(f"Scoring complete, generated {len(df)} rows of output")
    return df_preview, str(out_file)


# ---------------------------------------------------------------------
# Gradio UI Builder
# ---------------------------------------------------------------------

# -------------------------------------------------------------------- UI helpers
def _banner() -> None:
    gr.Markdown(BANNER_MD)


def _about_content() -> str:
    return (
        "## Welcome to the UCI Phonotactic Calculator!\n\n"
        "This is a research tool that allows users to calculate a variety of **phonotactic metrics**. "
        "These metrics are intended to capture how probable a word is based on the sounds it contains "
        "and the order in which those sounds are sequenced.\n\n"
        "For example, a nonce word like [stik] 'steek' might have a relatively high phonotactic score in English "
        "even though it is not a real word, because there are many words that begin with [st], end with [ik], and so on. "
        "In Spanish, however, this word would have a low score because there are no Spanish words that begin with the sequence [st]. "
        "A sensitivity to the phonotactic constraints of one's language(s) is an important component of linguistic competence, "
        "and the various metrics computed by this tool instantiate different models of how this sensitivity is operationalized.\n\n"
        "### The general use case for this tool is as follows:\n\n"
        "1. **Choose a training file.** You can either upload your own or choose one of the default training files "
        "(see the About page for details on how these should be formatted and the Datasets page for a description of the default files). "
        "This file is intended to represent the input over which phonotactic generalizations are formed, "
        "and will typically be something like a dictionary (a large list of word types). "
        "The models used to calculate the phonotactic metrics will be fit to this data.\n\n"
        "2. **Upload a test file.** The trained models will assign scores for each metric to the words in this file. "
        "This file may duplicate data in the training file (if you are interested in the scores assigned to existing words) "
        "or not (if you are interested in the predictions the various models make about how speakers generalize to new forms).\n\n"
        "The calculator computes a suite of metrics that are based on unigram/bigram frequencies "
        "(that is, the frequencies of individual sounds and the frequencies of adjacent pairs of sounds). "
        "This includes type- and token-weighted variants of the positional unigram/bigram method from Jusczyk et al. (1994) "
        "and Vitevitch and Luce (2004), as well as type- and token-weighted variants of standard unigram/bigram probabilities. "
        "See the About page for a detailed description of how these models differ and how to interpret the scores.\n\n"
        "### Citing the UCI Phonotactic Calculator\n"
        "If you publish work that uses the UCI Phonotactic Calculator, please cite the GitHub repository:\n\n"
        "> Mayer, C., Kondur, A., & Sundara, M. (2022). UCI Phonotactic Calculator (Version 0.1.0) [Computer software]. https://doi.org/10.5281/zenodo.7443706"
    )


def _about_box() -> None:
    with gr.Accordion("ℹ️ What is this tool?", open=False):
        gr.Markdown(_about_content())


def _footer() -> None:
    import uci_phonotactic_calculator as upc

    gr.Markdown(
        f"<footer>UCI Phonotactic Calculator • v{upc.__version__} • "
        "<a href='https://github.com/connormayer/uci_phonotactic_calculator' "
        "target='_blank'>GitHub</a></footer>"
    )


def _spacer(px: int = 8) -> None:
    gr.Markdown(f"<div style='height:{px}px'></div>")


# -------------------------------------------------------------------- TAB 1 – Calculator
def _calculator_tab():
    """Build the main calculator tab with a four-panel layout"""
    # Dictionary to store UI components for sharing with other tabs
    components = {}

    # Define available training files
    file_choices = [
        ("English", "english.csv"),
        ("English with frequencies", "english_freq.csv"),
        ("English needle words", "english_needle.csv"),
        ("English onsets", "english_onsets.csv"),
        ("Finnish", "finnish.csv"),
        ("French", "french.csv"),
        ("Polish onsets", "polish_onsets.csv"),
        ("Samoan", "samoan.csv"),
        ("Spanish with stress", "spanish_stress.csv"),
        ("Turkish", "turkish.csv"),
    ]

    gr.Markdown("<div class='feature-box'><strong>UCI Phonotactic Calculator</strong></div>")

    # Four-panel grid layout
    with gr.Row():
        # Top row - Training Data and Test Data
        with gr.Column(scale=1):
            # Panel 1: Training Data Selection
            with gr.Group(elem_classes=["input-card"]):
                gr.HTML('<h4 class="card-title">📁 Training Data</h4>')
                gr.HTML('<p class="card-subtitle">Select training corpus</p>')
                
                # Built-in datasets dropdown
                components["default_file"] = gr.Dropdown(
                    label="Built-in Datasets",
                    choices=[value for _, value in file_choices],
                    interactive=True,
                    elem_classes=["dropdown-style-upload"]
                )

                gr.HTML('<div class="option-divider"><span>OR</span></div>')

                # Custom training files upload (styled as dropdown)
                components["train_in"] = gr.UploadButton(
                    label="📁 Upload CSV",
                    file_types=[".csv"],
                    file_count="single",
                    elem_classes=["dropdown-style-upload", "upload-csv-button"]
                )
                gr.HTML('<p class="text-small text-muted">Click to browse and upload CSV file</p>')

                # Demo pair option
                components["use_demo_pair"] = gr.Checkbox(
                    label="🔄 Use complete English demo",
                    value=False
                )
                gr.HTML('<p class="text-small text-muted">Use demo training + test files</p>')

        with gr.Column(scale=1):
            # Panel 2: Test Data Upload
            with gr.Group(elem_classes=["input-card"]):
                gr.HTML('<h4 class="card-title">📊 Test Data</h4>')
                gr.HTML('<p class="card-subtitle">Upload words to score</p>')
                
                components["test_in"] = gr.File(
                    label="Upload test CSV file",
                    file_types=[".csv"],
                    interactive=True
                )
                gr.HTML('<p class="text-small text-muted">CSV with test words (one column)</p>')

    with gr.Row():
        # Bottom row - Model Configuration and Calculation
        with gr.Column(scale=1):
            # Panel 3: Model Configuration
            with gr.Group(elem_classes=["input-card"]):
                gr.HTML('<h4 class="card-title">⚙️ Model Settings</h4>')
                gr.HTML('<p class="card-subtitle">Configure calculation parameters</p>')
                
                # Model type selection
                components["model"] = gr.Radio(
                    choices=["ngram", "neighbourhood"],
                    value="ngram",
                    label="Model Type"
                )
                gr.HTML('<p class="text-small text-muted">Choose phonotactic model</p>')
                
                # N-gram order (shown for ngram model)
                components["ngram_order"] = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="N-gram Order"
                )
                gr.HTML('<p class="text-small text-muted">Higher = longer patterns</p>')
                
                # Additional options
                components["run_full_grid"] = gr.Checkbox(
                    label="Run full parameter grid",
                    value=False
                )

        with gr.Column(scale=1):
            # Panel 4: Advanced Options & Calculation
            with gr.Group(elem_classes=["input-card"]):
                gr.HTML('<h4 class="card-title">🎛️ Advanced Options</h4>')
                gr.HTML('<p class="card-subtitle">Fine-tune calculation settings</p>')
                
                # Weight mode
                gr.HTML('<p class="option-label">Weight Mode</p>')
                components["weight_mode"] = gr.Radio(
                    ["None", "Raw", "Log"], 
                    value="None", 
                    label=""
                )
                gr.HTML('<p class="text-small text-muted">Frequency weighting</p>')

                # Probability mode
                gr.HTML('<p class="option-label">Probability Mode</p>')
                components["prob_mode"] = gr.Radio(
                    ["Joint", "Conditional"], 
                    value="Joint", 
                    label=""
                )
                gr.HTML('<p class="text-small text-muted">Calculation method</p>')

                # Manual filters
                components["filter_string"] = gr.Textbox(
                    label="Advanced Filters",
                    placeholder="e.g., smoothing=add1",
                    lines=1
                )

                # Progress option
                components["hide_progress"] = gr.Checkbox(
                    label="Hide progress details",
                    value=False
                )

    # Calculation button - full width below panels
    with gr.Row():
        with gr.Column():
            gr.HTML('<div style="margin: 20px 0 10px 0;"></div>')  # Spacer
            components["calculate_btn"] = gr.Button(
                "🚀 Calculate Phonotactic Scores",
                variant="primary",
                size="lg",
                elem_classes=["calculate-button"]
            )

    # Results Section (initially hidden) - below everything
    components["results_container"] = gr.Group(visible=False, elem_classes=["results-container"])
    with components["results_container"]:
        gr.HTML('<div style="margin: 30px 0 20px 0;"><h4 class="card-title">📈 Results</h4></div>')
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<h5 style="margin: 0 0 10px 0; color: #2c3e50;">Preview</h5>')
                components["results_df"] = gr.Dataframe(
                    interactive=False,
                    wrap=True,
                    elem_classes=["results-table"]
                )
            
            with gr.Column(scale=1):
                gr.HTML('<h5 style="margin: 0 0 10px 0; color: #2c3e50;">Download</h5>')
                with gr.Group(elem_classes=["download-section"]):
                    components["download_file"] = gr.File(
                        label=None,
                        interactive=False,
                        elem_classes=["download-file"]
                    )
                    components["download_label"] = gr.HTML("")

    gr.Markdown("</div>")
    
    # Interactive behavior
    def update_ui_for_model(model_name):
        """Show/hide model-specific options"""
        if model_name == "ngram":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)

    components["model"].change(
        fn=update_ui_for_model,
        inputs=[components["model"]],
        outputs=[components["ngram_order"]]
    )

    # Build filter string from selections
    def build_filter_string(*args):
        """Combine radio button selections into filter string"""
        weight_mode, prob_mode, manual_filter = args
        
        filters = []
        if weight_mode and weight_mode != "None":
            filters.append(f"weight_mode={weight_mode.lower()}")
        if prob_mode:
            filters.append(f"prob_mode={prob_mode.lower()}")
        
        auto_filter = " ".join(filters)
        
        # Combine with manual filter if provided
        if manual_filter and manual_filter.strip():
            if auto_filter:
                return f"{auto_filter} {manual_filter.strip()}"
            else:
                return manual_filter.strip()
        
        return auto_filter

    # Main calculation function
    def process_and_score(*args):
        """Process inputs and run the scoring calculation"""
        try:
            # Extract arguments
            (default_file, train_file, use_demo_pair, test_file, model, 
             ngram_order, run_full_grid, weight_mode, prob_mode, 
             manual_filter, hide_progress) = args
            
            # Handle file selection logic
            actual_train_file = None
            actual_test_file = test_file
            
            if use_demo_pair:
                # Use demo files - set specific demo files instead of None
                actual_train_file = "english.csv"  # Default English training file
                actual_test_file = "sample_test_data/english_test_data.csv"  # Correct demo test file path
                logger.info("Using demo pair: english.csv and sample_test_data/english_test_data.csv")
            else:
                # Handle training file selection
                if default_file and default_file.strip():
                    # Use the selected default file name
                    actual_train_file = default_file
                    logger.info(f"Using default training file: {default_file}")
                elif train_file is not None:
                    # Use uploaded custom training file
                    actual_train_file = train_file
                    logger.info(f"Using uploaded training file: {train_file}")
                else:
                    raise gr.Error("Please select a default training file OR upload a custom training file")
                
                # Test file is required when not using demo pair
                if test_file is None:
                    raise gr.Error("Please upload a test file")
                
                actual_test_file = test_file
            
            # Build complete filter string
            filter_str = build_filter_string(weight_mode, prob_mode, manual_filter)
            
            # Run the calculation with correct parameters
            df, csv_path = score(
                train_csv=actual_train_file,
                test_csv=actual_test_file,
                model=model,
                run_full_grid=run_full_grid,
                ngram_order=ngram_order,
                filter_string=filter_str,
                hide_progress=hide_progress
            )
            
            # Create download label
            filename = Path(csv_path).name if csv_path else "results.csv"
            download_label = f'<p style="margin:0;color:#2c3e50;font-weight:500;background-color:#f8f9fa;padding:8px;border-radius:4px;border:1px solid #dee2e6;">📄 {filename}</p>'
            
            return (
                gr.update(visible=True),  # Show results container
                df,  # Results dataframe
                csv_path,  # Download file
                download_label  # Download label
            )
            
        except Exception as e:
            # Show error in results
            error_msg = f"❌ Error: {str(e)}"
            return (
                gr.update(visible=True),
                None,
                None,
                f'<p style="margin:0;color:#ff6b6b;">{error_msg}</p>'
            )

    # Wire up the calculate button
    components["calculate_btn"].click(
        fn=process_and_score,
        inputs=[
            components["default_file"],
            components["train_in"],
            components["use_demo_pair"],
            components["test_in"],
            components["model"],
            components["ngram_order"],
            components["run_full_grid"],
            components["weight_mode"],
            components["prob_mode"],
            components["filter_string"],
            components["hide_progress"]
        ],
        outputs=[
            components["results_container"],
            components["results_df"],
            components["download_file"],
            components["download_label"]
        ]
    )

    return components


# -------------------------------------------------------------------- TAB 2 – Datasets
def _datasets_tab(components=None):
    gr.Markdown("<div class='feature-box'><strong>Available Datasets</strong></div>")

    with gr.Accordion("Built-in Demo Datasets", open=True):
            gr.Markdown(
                """
UCI Phonotactic Calculator
File List
english.csv
A subset of the CMU Pronouncing Dictionary with CELEX frequencies > 1. This is notated in ARPABET. Numbers indicating vowel stress have been removed.
english_freq.csv
A subset of the CMU Pronouncing Dictionary with CELEX frequencies. This data is represented in ARPABET.
english_needle.csv
Data set from Needle et al. (2022). Consists of about 11,000 monomorphemic words from CELEX (Baayen et al. 1995) in ARPABET transcription.
english_onsets.csv
55 English onsets and their CELEX type frequencies in ARPABET format from Hayes & Wilson (2008). A subset of the onsets in the CMU Pronouncing Dictionary.
finnish.csv
From a word list generated by the Institute for the Languages of Finland (http://kaino.kotus.fi/sanat/nykysuomi/). Represented orthographically. See Mayer (2020) for details.
french.csv
French corpus used in Goldsmith & Xanthos (2009) and Mayer (2020). Represented in IPA.
polish_onsets.csv
Polish onsets with type frequencies from Jarosz (2017). Generated from a corpus of child-directed speech consisting of about 43,000 word types (Haman et al. 2011). Represented orthographically.
samoan.csv
Samoan word list from Milner (1993), compiled by Kie Zuraw. Represented in IPA.
spanish_stress.csv
A set of about 24,000 word types including inflected forms from the EsPal database (Duchon et al. 2013) in IPA with stress encoded. Frequencies from a large collection of Spanish subtitle data.
turkish.csv
A set of about 18,000 citation forms from the Turkish Electronic Living Lexicon database (TELL; Inkelas et al. 2000) in IPA.
"""
            )

    with gr.Accordion("Using Your Own Data", open=True):
        gr.Markdown(
            "To use your own data, uncheck the 'Use built-in demo data' checkbox in the Calculator tab and upload your own CSV files.\n\n"
            "#### Required format\n"
            "- Both the training and the test files must be in CSV format (.csv)\n"
            "- Both the training and test files must be in CSV format (.csv)\n"
            "- The training file should consist of one or two columns with no headers\n"
            "  * First column (mandatory): Word list with space-separated symbols\n"
            "  * Second column (optional): Word frequencies as raw counts\n"
            "- The test file should consist of a single column containing the test word list\n"
            "- The output file will contain one column containing the test words, one column containing the number of symbols in the word, "
            "and one column for each of the metrics.\n\n"
            "For more detailed information about the expected format, please see the About tab."
        )

    with gr.Accordion("Example Datasets", open=True):
        gr.Markdown(
            "Below are examples of correctly formatted data for different languages and purposes.\n"
            "### English Example (english_freq.csv)\n"
            "```\n"
            "c a t, 42\n"
            "d o g, 37\n"
            "f i s h, 18\n"
            "```\n\n"
            "### IPA Transcription Example\n"
            "```\n"
            "k æ t, 42\n"
            "d ɔ g, 37\n"
            "f ɪ ʃ, 18\n"
            "```\n\n"
            "### Finnish Example (finnish.csv)\n"
            "```\n"
            "k i s s a\n"
            "k o i r a\n"
            "k a l a\n"
            "```\n\n"
            "For additional example datasets and formatting guidelines, check the [GitHub repository](https://github.com/connormayer/uci_phonotactic_calculator)."
        )

    with gr.Accordion("Download Sample Data", open=False):
        gr.Markdown(
            "You can download sample datasets from the GitHub repository:\n"
            "[UCI Phonotactic Calculator Samples](https://github.com/connormayer/uci_phonotactic_calculator/tree/main/src/uci_phonotactic_calculator/data/sample_test_data)"
        )


# -------------------------------------------------------------------- TAB 3 – GitHub
def _github_tab(components=None):
    gr.Markdown("<div class='feature-box'><strong>GitHub Repository</strong></div>")

    with gr.Accordion("Repository Links", open=True):
        gr.Markdown(
            "### UCI Phonotactic Calculator on GitHub\n\n"
            "The full source code for this tool is available on GitHub:\n"
            "[github.com/connormayer/uci_phonotactic_calculator](https://github.com/connormayer/uci_phonotactic_calculator)\n\n"
            "#### Quick Links\n"
            "- [Documentation](https://github.com/connormayer/uci_phonotactic_calculator#readme)\n"
            "- [Sample Data](https://github.com/connormayer/uci_phonotactic_calculator/tree/main/src/uci_phonotactic_calculator/data)\n"
            "- [Issue Tracker](https://github.com/connormayer/uci_phonotactic_calculator/issues)\n"
        )

    with gr.Accordion("Features", open=True):
        gr.Markdown(
            "### Key Features\n"
            "- **Command-line interface**: Run calculations from your terminal\n"
            "- **Python API**: Integrate with your Python code\n"
            "- **Web interface**: This Gradio UI and Django-based version\n"
            "- **Extensible plugin system**: Add custom models and metrics\n"
            "- **Comprehensive documentation**: Detailed explanations of all metrics\n"
            "- **Cross-platform**: Works on Windows, macOS, and Linux\n"
        )

    with gr.Accordion("Installation", open=False):
        gr.Markdown(
            "### Installing from PyPI\n"
            "```bash\n"
            "pip install uci-phonotactic-calculator\n"
            "```\n\n"
            "### Installing from Source\n"
            "```bash\n"
            "git clone https://github.com/connormayer/uci_phonotactic_calculator.git\n"
            "cd uci_phonotactic_calculator\n"
            "pip install -e .\n"
            "```\n"
        )

    with gr.Accordion("Contributing", open=False):
        gr.Markdown(
            "### How to Contribute\n"
            "Contributions are welcome! Here's how you can help:\n\n"
            "1. **Report bugs**: Open an issue on GitHub if you find a bug\n"
            "2. **Request features**: Suggest new features or improvements\n"
            "3. **Add plugins**: Extend the calculator with new models or metrics\n"
            "4. **Improve documentation**: Help clarify or expand the documentation\n"
            "5. **Submit pull requests**: Fix bugs or add features directly\n\n"
            "Please see the [contributing guidelines](https://github.com/connormayer/uci_phonotactic_calculator/blob/main/CONTRIBUTING.md) for more information."
        )


# -------------------------------------------------------------------- TAB 4 – About
def _about_tab(components=None):
    gr.Markdown(
        "<div class='feature-box'><strong>About UCI Phonotactic Calculator</strong></div>"
    )

    with gr.Accordion("Data Format", open=True):
        gr.Markdown(
            "The simplest way to understand the format of the input data is to look at examples on the Datasets tab. Read below for more details:\n\n"
            "1. Both the training and the test file must be in comma-separated format (.csv).\n\n"
            "2. The training file should consist of one or two columns with no headers.\n"
            "   * The first column (mandatory) contains a word list, with each symbol (phoneme, orthographic letter, etc.) separated by spaces. "
            "For example, the word 'cat' represented in IPA would be \"k æ t\". You may use any transcription system or representation you like, "
            "so long as the individual symbols are separated by spaces. Because symbols are space-separated, they may be arbitrarily long: "
            "this allows the use of transcription systems like ARPABET, which use more than one character to represent individual sounds.\n"
            "   * The second column (optional) contains the corresponding frequencies for each word. These must be expressed as raw counts. "
            "These values are used in the token-weighted variants of the unigram and bigram models, which ascribe greater influence to the phonotactics of more frequent words. "
            "If this column is not provided, the token-weighted metrics will not be computed, but the other metrics will be returned.\n\n"
            "3. The test file should consist of a single column containing the test word list. The same format as the training file must be used.\n\n"
            "4. The output file will contain one column containing the test words, one column containing the number of symbols in the word, "
            "and one column for each of the metrics."
        )

    with gr.Accordion("Unigram/Bigram Scores", open=False):
        gr.Markdown(
            "The UCI Phonotactic Calculator currently supports a suite of unigram and bigram metrics that share the property of being "
            "sensitive only to the frequencies of individual sounds or adjacent pairs of sounds. Here is a summary of the columns in the output file:\n\n"
            "* `word`: The word\n"
            "* `word_len`: The number of symbols in the word\n"
            "* `uni_prob`: Unigram probability\n"
            "* `uni_prob_freq_weighted`: Frequency-weighted unigram probability\n"
            "* `uni_prob_smoothed`: Add-one smoothed unigram probability\n"
            "* `uni_prob_freq_weighted_smoothed`: Add-one smoothed, frequency-weighted unigram probability\n"
            "* `bi_prob`: Bigram probability\n"
            "* `bi_prob_freq_weighted`: Frequency-weighted bigram probability\n"
            "* `bi_prob_smoothed`: Add-one smoothed bigram probability\n"
            "* `bi_prob_freq_weighted_smoothed`: Add-one smoothed, frequency-weighted bigram probability\n"
            "* `pos_uni_score`: Positional unigram score\n"
            "* `pos_uni_score_freq_weighted`: Frequency-weighted positional unigram score\n"
            "* `pos_uni_score_smoothed`: Add-one smoothed positional unigram score\n"
            "* `pos_uni_score_freq_weighted_smoothed`: Add-one smoothed, frequency-weighted positional unigram score\n"
            "* `pos_bi_score`: Positional bigram score\n"
            "* `pos_bi_score_freq_weighted`: Frequency-weighted positional bigram score\n"
            "* `pos_bi_score_smoothed`: Add-one smoothed positional bigram score\n"
            "* `pos_bi_score_freq_weighted_smoothed`: Add-one smoothed, frequency-weighted positional bigram score\n\n"
            "These columns can be broken down into four broad classes:\n"
            "1. unigram probabilities (`uni_prob`, `uni_prob_freq_weighted`, `uni_prob_smoothed`, `uni_prob_freq_weighted_smoothed`)\n"
            "2. bigram probabilities (`bi_prob`, `bi_prob_freq_weighted`, `bi_prob_smoothed`, `bi_prob_freq_weighted_smoothed`)\n"
            "3. positional unigram scores (`pos_uni_score`, `pos_uni_score_freq_weighted`, `pos_uni_score_smoothed`, `pos_uni_score_freq_weighted_smoothed`)\n"
            "4. positional bigram scores (`pos_bi_score`, `pos_bi_score_freq_weighted`, `pos_bi_score_smoothed`, `pos_bi_score_freq_weighted_smoothed`)\n\n"
            "Each of these classes has *frequency-weighted* and *smoothed* variants.\n"
            "* Frequency-weighted (or token-weighted) variants weight the occurrence of each unigram/bigram or positional unigram/bigram by the log token frequency of the word type it appears in. "
            "This effectively means that sound sequences in high frequency words 'count for more' than sound sequences in low-frequency words.\n"
            "* Smoothed variants assign a small part of the total share of probability to unseen configurations by assigning them pseudo-counts of 1 (add-one smoothing). "
            "For example, in an unsmoothed bigram probability model, any word that contains a bigram not found in the corpus data will be assigned a probability of 0. "
            "In the smoothed model, it will be assigned a low probability as though it had been observed once in the training data. "
            "Note that smoothed models will still assign zero probabilities if the training data contains any symbols not observed in the test data."
        )

    with gr.Accordion("Specific Metrics", open=False):
        gr.Markdown(
            "#### Unigram probability (`uni_prob`)\n\n"
            "This is the standard unigram probability where the probability of a word is the product of the probability of its individual symbols. "
            "Note that the probability of the individual symbols is based only on their frequency of occurrence, not the position in which they occur.\n\n"
            "If the test data contains symbols that do not occur in the training data, the tokens containing them will be assigned probabilities of 0.\n\n"
            "#### Bigram probability (`bi_prob`)\n\n"
            "This is the standard bigram probability where the probability of a word is the product of the probability of all the bigrams it contains. "
            "Note that the probability of the bigrams is based only on their frequency of occurrence, not the position in which they occur or their sequencing with respect to one another.\n\n"
            "Each word is padded with a special start and end symbol, which allows us to calculate bigram probabilities for symbols that begin and end words.\n\n"
            "#### Positional unigram score (`pos_uni_prob`)\n\n"
            "This is a type-weighted variant of unigram score from Vitevitch and Luce (2004).\n\n"
            "Under this metric, the score assigned to a word is based on the sum of the probability of its individual symbols occuring at their respective positions. "
            "Note that the ordering of the symbols with respect to one another does not affect the score, only their relative frequencies within their given positions. "
            "Higher scores represent words with more probable phonotactics, but note that this score cannot be interpreted as a probability.\n\n"
            "Vitevitch and Luce (2004) add 1 to the sum of the unigram probabilities 'to aid in locating these values when you cut and paste the output in the right field to another program.' "
            "They recommend subtracting 1 from these values before reporting them.\n\n"
            "#### Positional bigram score (`pos_bi_prob`)\n\n"
            "This is a type-weighted variant of the bigram score from Vitevitch and Luce (2004).\n\n"
            "Under this metric, the score assigned to a word is based on the sum of the probability of each contiguous pair of symbols occuring at their respective positions. "
            "Higher scores represent words with more probable phonotactics, but note that this score cannot be interpreted as a probability.\n\n"
            "Vitevitch and Luce (2004) add 1 to the sum of the bigram probabilities 'to aid in locating these values when you cut and paste the output in the right field to another program.' "
            "They recommend subtracting 1 from these values before reporting them.\n\n"
            "#### Token-weighted variants\n\n"
            "Assuming that the training data consists of a list of word types (e.g., a dictionary), the above metrics can be described as *type-weighted*: "
            "the frequency of individual word types has no bearing on the scores assigned by the metrics.\n\n"
            "The calculator also includes *token-weighted* variants of each of the above measures, where the phonotactic properties of frequent word types are weighted higher than those in less frequent word types. "
            "These are included under all the column names containing `freq_weighted`.\n\n"
            "These measures are computed by changing the count function such that it is the number of occurrences of the configuration in question multiplied by the natural log of the count of the word containing each occurrence.\n\n"
            'For example, suppose we have a corpus containing two word types "kæt", which occurs 1000 times, and "tæk", which occurs 50 times. '
            "Under a token-weighted unigram model, C(æ) = ln(1000) + ln(50) ≈ 10.82, while in a type-weighted unigram model C(æ) = 1 + 1 = 2.\n\n"
            "The token-weighted positional ungiram and bigram scores correspond to the metrics presented in Vitevitch and Luce (2004), though they use the base-10 logarithm rather than the natural logarithm.\n\n"
            "#### Smoothing\n\n"
            "The calculator also includes add-one smoothed (or Laplace Smoothed) variants of each measure.\n\n"
            "Under add-one smoothing, each configuration we could (unigrams, bigrams, positional unigrams, positional bigrams) begins with a default count of 1, rather than 0. "
            "This means that configurations that are not observed in the training data are treated as though they have been observed once, which gives them a small, rather than zero, probability. "
            "This effectively spreads some of the probability mass from attested configurations onto unattested ones.\n\n"
            "Smoothing in these models assigns non-zero probabilities to unattested sequences of known symbols, but not to unknown symbols (which is why there is no smoothing for unigram probabilities). "
            "Any words in the test data containing symbols not found in the training data are assigned probabilities of zero.\n\n"
            "In the token-weighted versions of the metrics, smoothing is also done by adding one to the log-weighted counts."
        )

    with gr.Accordion("References", open=False):
        gr.Markdown(
            "Vitevitch, M.S., & Luce, P.A. (2004). A web-based interface to calculate phonotactic probability for words and nonwords in English. "
            "*Behavior Research Methods, Instruments, & Computers, 36*(3), 481-487."
        )


# -------------------------------------------------------------------- TAB 5 – Documentation
def _docs_tab(components=None):
    gr.Markdown("<div class='feature-box'><strong>Documentation</strong></div>")

    # Find the README file
    import os
    from pathlib import Path
    import pkg_resources

    # Get the root directory of the project
    try:
        # Try to find the README in the package
        package_path = Path(
            pkg_resources.resource_filename("uci_phonotactic_calculator", "")
        )
        root_dir = package_path.parent.parent
        readme_path = root_dir / "README.md"

        if not readme_path.exists():
            # If not found, try a relative path from the current file
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent.parent
            readme_path = project_root / "README.md"

        # Load the README content
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()

            # Display the README content
            gr.Markdown(readme_content)
        else:
            gr.Markdown(
                "### Documentation\n\n"
                "The README file could not be found. Please visit the GitHub repository for documentation:\n"
                "[GitHub Repository](https://github.com/connormayer/uci_phonotactic_calculator)"
            )
    except Exception as e:
        # Fallback if there's an error loading the README
        gr.Markdown(
            "### Documentation\n\n"
            f"Error loading documentation: {str(e)}\n\n"
            "Please visit the GitHub repository for documentation:\n"
            "[GitHub Repository](https://github.com/connormayer/uci_phonotactic_calculator)"
        )


# -------------------------------------------------------------------- MAIN builder
def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title=APP_TITLE) as demo:

        _banner()
        _about_box()


        with gr.Tabs() as tabs:
            with gr.TabItem("📊 Calculator", id=0) as tab_main:
                components = _calculator_tab()
            with gr.TabItem("📂 Datasets", id=1):
                _datasets_tab()
            with gr.TabItem("🔗 GitHub", id=2):
                _github_tab()
            with gr.TabItem("ℹ️ About", id=3):
                _about_tab()
            with gr.TabItem("📚 Documentation", id=4):
                _docs_tab(None)  # Pass None since we're not using components

        _footer()

    return demo


def main():
    """Main entry point to run the web UI with clear console output."""
    print("\n========================================")
    print("UCI Phonotactic Calculator - Web UI")
    print("========================================\n")
    print("Starting web server and opening browser automatically...")
    demo = build_ui()
    demo.queue(max_size=10)
    # Launch with inbrowser=True to automatically open the browser
    demo.launch(inbrowser=True, show_error=True)


# Direct script execution
if __name__ == "__main__":
    main()
