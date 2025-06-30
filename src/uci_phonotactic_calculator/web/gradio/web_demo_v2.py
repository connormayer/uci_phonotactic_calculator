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
from typing import Any, Dict, List, Optional, Tuple, cast

from gradio.blocks import Block
from gradio.components.base import Component

# NEW ▶
CSS = """
/* Main container and layout */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
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

/* Equal height step cards */
.step-card {
    background: white;
    border: 1px solid #e1e5e9;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.step-card:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-color: #667eea;
}

.step-card h3 {
    font-size: 16px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.step-card ul {
    flex: 1;
    margin: 0;
}

/* Results table scrolling support */
.results-table {
    max-height: 400px !important;
    overflow-y: auto !important;
    overflow-x: auto !important;
}

.results-table > div {
    max-height: 400px !important;
    overflow-y: auto !important;
    overflow-x: auto !important;
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
def _calculator_tab() -> Dict[str, Block]:
    """Build the main calculator tab with a four-panel layout"""
    # Dictionary to store UI components for sharing with other tabs
    components: Dict[str, Block] = {}

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

    gr.Markdown("<div class='feature-box'><strong>🧮 UCI Phonotactic Calculator</strong></div>")

    # Quick Start Guide
    with gr.Accordion("🚀 Quick Start Guide", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h3>📁 Step 1: Choose Your Data</h3>
                <ul>
                <li><strong>Upload your own files</strong>: Use CSV files with a 'word' column</li>
                <li><strong>Try the demo</strong>: Click "Load Demo Data" for sample English words</li>
                </ul>
                </div>
                """)
                
                gr.Markdown("""
                <div class='step-card'>
                <h3>⚙️ Step 3: Configure Settings</h3>
                <ul>
                <li><strong>N-gram Order</strong>: How many characters to consider (2-4 recommended)</li>
                <li><strong>Advanced Options</strong>: Fine-tune probability calculations</li>
                </ul>
                </div>
                """)
            
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h3>🔧 Step 2: Select a Model</h3>
                <ul>
                <li><strong>N-gram Model</strong>: Classic phonotactic probability</li>
                <li><strong>Positional N-gram</strong>: Position-aware analysis (more sophisticated)</li>
                </ul>
                </div>
                """)
                
                gr.Markdown("""
                <div class='step-card'>
                <h3>📊 Step 4: Calculate & Download</h3>
                <ul>
                <li>Click "Calculate Phonotactic Scores" to process your data</li>
                <li>View results in the preview table</li>
                <li>Download the complete scored dataset as CSV</li>
                </ul>
                </div>
                """)

    # Data Selection Section
    with gr.Accordion("📁 Data Selection", open=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
<div class='step-card'>
<h4>📚 1. Training Data</h4>
<p>Choose the corpus to train your phonotactic model</p>
</div>
                """)
                
                # Built-in datasets dropdown
                components["default_file"] = gr.Dropdown(
                    label="Built-in Datasets",
                    choices=[value for _, value in file_choices],
                    interactive=True,
                    elem_classes=["dropdown-style-upload"]
                )

                gr.HTML('<div class="option-divider"><span>OR</span></div>')

                # Custom training files upload
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
            
            with gr.Column():
                gr.Markdown("""
<div class='step-card'>
<h4>🎯 2. Test Data</h4>
<p>Upload the words you want to score</p>
</div>
                """)
                
                components["test_in"] = gr.File(
                    label="Upload test CSV file",
                    file_types=[".csv"],
                    interactive=True
                )
                gr.HTML('<p class="text-small text-muted">CSV with test words (one column)</p>')

    # Model Configuration Section
    with gr.Accordion("⚙️ Model Configuration", open=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
<div class='step-card'>
<h4>🔧 Model Settings</h4>
<p>Configure the phonotactic calculation parameters</p>
</div>
                """)
                
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
                    label="Run full parameter grid, all variations (may take ~20 minutes)",
                    value=False
                )
            
            with gr.Column():
                gr.Markdown("""
<div class='step-card'>
<h4>🎛️ Advanced Options</h4>
<p>Fine-tune calculation settings</p>
</div>
                """)
                
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

    # Calculation Section
    with gr.Accordion("🚀 Calculate", open=True):
        gr.Markdown("""
<div class='input-card'>
<h3>▶️ Run Calculation</h3>
<p>Click the button below to calculate phonotactic scores based on your selected data and model settings.</p>
</div>
        """)
        
        components["calculate_btn"] = gr.Button(
            "🚀 Calculate Phonotactic Scores",
            variant="primary",
            size="lg",
            elem_classes=["calculate-button"]
        )

    # Results Section (initially hidden)
    components["results_container"] = cast(
        Any, gr.Group(visible=False, elem_classes=["results-container"])
    )
    with cast(gr.Group, components["results_container"]):
        gr.Markdown("""
<div class='feature-box'>
<strong>📈 Results</strong>
</div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<h5 style="margin: 0 0 10px 0; color: #2c3e50;">Preview</h5>')
                components["results_df"] = gr.Dataframe(
                    interactive=False,
                    wrap=True,
                    row_count=(20, "paginate")
                )
            
            with gr.Column(scale=1):
                gr.HTML('<h5 style="margin: 0 0 10px 0; color: #2c3e50;">Download</h5>')
                with gr.Group(elem_classes=["download-section"]):
                    components["download_label"] = gr.HTML(
                        '<p style="color: #57606a; font-size: 0.9em;">Ready to calculate.</p>'
                    )
                    components["download_file"] = gr.File(
                        label="Download Results (CSV)",
                        interactive=False,
                        visible=False,  # Initially hidden
                    )

    # Interactive behavior
    def _safe_remove(path: str) -> None:
        """Safely remove a file if it exists."""
        if os.path.exists(path):
            os.remove(path)

    def update_ui_for_model(model_name: str) -> dict[str, Any]:
        """Show/hide model-specific options"""
        is_ngram = model_name.lower() == "ngram"
        return gr.update(visible=is_ngram)

    cast(gr.Radio, components["model"]).change(
        fn=update_ui_for_model,
        inputs=[components["model"]],
        outputs=[components["ngram_order"]],
    )

    # Main calculation function
    def process_and_score(
        default_file: str,
        train_file: Optional[str],
        use_demo_pair: bool,
        test_file: Optional[str],
        model: str,
        ngram_order: int,
        run_full_grid: bool,
        weight_mode: str,
        prob_mode: str,
        manual_filter: str,
    ) -> Tuple[dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]]:
        """Process inputs, run the scoring calculation, and update UI components."""
        logger.info("=== GRADIO CALCULATION START ===")
        for name, value in locals().items():
            if name != "request":
                logger.info(f"{name}: {value}")

        actual_train_file, actual_test_file = None, None
        if use_demo_pair:
            demo_train, demo_test = get_demo_paths(default_file)
            logger.info(f"Using demo pair: {demo_train} and {demo_test}")
            actual_train_file, actual_test_file = str(demo_train), str(demo_test)
        else:
            actual_train_file = train_file
            actual_test_file = test_file

        if not actual_train_file or not actual_test_file:
            msg = "Missing training or test file. Please select or upload both."
            logger.warning(msg)
            gr.Warning(msg)
            return (
                gr.update(visible=False),
                pd.DataFrame(),
                gr.update(visible=False),
                gr.update(value=f'<p style="color: #d9534f;">⚠️ {msg}</p>'),
            )

        extra_args: List[str] = []
        filters_dict: Dict[str, Any] = {}

        if not run_full_grid:
            extra_args += ["-n", str(ngram_order)]
            if weight_mode and weight_mode != "None":
                extra_args += ["--weight-mode", weight_mode.lower()]
            if prob_mode:
                extra_args += ["--prob-mode", prob_mode.lower()]

        if manual_filter:
            try:
                pairs = [p.strip().split("=") for p in manual_filter.split(",")]
                filters_dict.update({k.strip(): v.strip() for k, v in pairs})
            except Exception:
                gr.Warning("Invalid manual filter format. Use 'key=value, key2=value2'.")

        try:
            out_path = os.path.join(tempfile.gettempdir(), f"scores_{uuid.uuid4().hex}.csv")
            # Register a cleanup function that is compatible with all Python versions
            atexit.register(functools.partial(_safe_remove, out_path))

            logger.info(f"Running calculation with extra_args: {extra_args}")
            ngram_run(
                train_file=actual_train_file,
                test_file=actual_test_file,
                output_file=out_path,
                model=None if run_full_grid else model,
                run_all=run_full_grid,
                filters=filters_dict,
                show_progress=False,
                extra_args=extra_args,
            )

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                df = pd.read_csv(out_path)
                logger.info(f"DataFrame shape: {df.shape}")
                if df.empty:
                    msg = "Calculation ran but produced no output rows."
                    gr.Warning(msg)
                    return (
                        gr.update(visible=False),
                        pd.DataFrame(),
                        gr.update(visible=False),
                        gr.update(value=f'<p style="color: #f0ad4e;">⚠️ {msg}</p>'),
                    )

                status_msg = f"✅ Success! Previewing {len(df)} rows."
                return (
                    gr.update(visible=True),
                    df,
                    gr.update(value=out_path, visible=True),
                    gr.update(value=f'<p style="color: #5cb85c;">{status_msg}</p>'),
                )
            else:
                msg = "Calculation failed: Output file not created or is empty."
                logger.error(msg)
                gr.Error(msg)
                df = pd.DataFrame()

        except Exception as e:
            msg = f"An error occurred: {e}"
            logger.exception(f"An error occurred during calculation: {e}")
            gr.Error(msg)
            df = pd.DataFrame()

        return (
            gr.update(visible=False),
            df,
            gr.update(visible=False),
            gr.update(value=f'<p style="color: #d9534f;">❌ {msg}</p>'),
        )

    # Wire up the calculate button
    cast(gr.Button, components["calculate_btn"]).click(
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
        ],
        outputs=[
            components["results_container"],
            components["results_df"],
            components["download_file"],
            components["download_label"],
        ],
    )

    return components


# -------------------------------------------------------------------- TAB 2 – Datasets
def _datasets_tab() -> None:
    gr.Markdown("<div class='feature-box'><strong>📊 Available Datasets</strong></div>")

    # Built-in datasets section
    with gr.Accordion("🗃️ Built-in Demo Datasets", open=True):
        gr.Markdown("""
        <div class='input-card'>
        <h3>📋 Dataset Overview</h3>
        <p>The calculator includes several pre-loaded datasets for testing and demonstration purposes. These cover multiple languages and transcription systems.</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h4>🇺🇸 English Datasets</h4>
                <ul>
                <li><strong>english.csv</strong> - CMU Dictionary subset (ARPABET)</li>
                <li><strong>english_freq.csv</strong> - With CELEX frequencies</li>
                <li><strong>english_needle.csv</strong> - Needle et al. (2022) dataset</li>
                <li><strong>english_onsets.csv</strong> - Hayes & Wilson (2008) onsets</li>
                </ul>
                </div>
                
                <div class='step-card'>
                <h4>🌍 European Languages</h4>
                <ul>
                <li><strong>finnish.csv</strong> - Finnish orthographic words</li>
                <li><strong>french.csv</strong> - French corpus (IPA)</li>
                <li><strong>polish_onsets.csv</strong> - Polish onsets</li>
                <li><strong>turkish.csv</strong> - Turkish lexicon (IPA)</li>
                </ul>
                </div>
                """)
            
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h4>🗺️ Other Languages</h4>
                <ul>
                <li><strong>samoan.csv</strong> - Samoan word list (IPA)</li>
                <li><strong>spanish_stress.csv</strong> - Spanish with stress</li>
                </ul>
                </div>
                
                <div class='step-card'>
                <h4>📈 Usage Tips</h4>
                <ul>
                <li>Try <strong>english.csv</strong> for your first test</li>
                <li>Use <strong>english_freq.csv</strong> for frequency-weighted metrics</li>
                <li>Different transcription systems available</li>
                </ul>
                </div>
                """)

    # Using your own data section
    with gr.Accordion("📁 Using Your Own Data", open=False):
        gr.Markdown("""
        <div class='input-card'>
        <h3>💡 Quick Start</h3>
        <p>To use your own data, simply uncheck <strong>"Use built-in demo data"</strong> in the Calculator tab and upload your CSV files.</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h4>📋 Training File Requirements</h4>
                <ul>
                <li><strong>Format:</strong> CSV file with no headers</li>
                <li><strong>Column 1:</strong> Word list with symbols separated by spaces<br>
                    <em>Example: "cat" becomes "k æ t" in IPA</em></li>
                <li><strong>Column 2 (Optional):</strong> Raw frequency counts</li>
                </ul>
                <p><em>Example: "c a t, 42"</em></p>
                </div>
                """)
            
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h4>🎯 Test File Requirements</h4>
                <ul>
                <li><strong>Format:</strong> Single column CSV</li>
                <li><strong>Same transcription:</strong> Match training file</li>
                <li><strong>Space-separated symbols</strong> (e.g., "d ɔ g" for "dog")</li>
                </ul>
                <p><em>Example: "d ɔ g"</em></p>
                </div>
                """)

    # Example datasets section
    with gr.Accordion("📝 Formatting Examples", open=False):
        gr.Markdown("""
        <div class='input-card'>
        <h3>✨ Example Formats</h3>
        <p>Here are examples of correctly formatted data for different languages and transcription systems:</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
<div class='step-card'>
<h4>🔤 English with Frequencies</h4>
<pre><code>c a t, 42
d o g, 37
f i s h, 18</code></pre>
<p><em>Training file with orthographic symbols and frequency counts</em></p>
</div>

<div class='step-card'>
<h4>🔊 IPA Transcription</h4>
<pre><code>k æ t, 42
d ɔ g, 37
f ɪ ʃ, 18</code></pre>
<p><em>Training file using International Phonetic Alphabet</em></p>
</div>
                """)
            
            with gr.Column():
                gr.Markdown("""
<div class='step-card'>
<h4>🇫🇮 Finnish Example</h4>
<pre><code>k i s s a
k o i r a
k a l a</code></pre>
<p><em>Simple word list without frequencies (orthographic)</em></p>
</div>

<div class='step-card'>
<h4>🎯 Test File Example</h4>
<pre><code>n e w w o r d
t e s t w o r d
a n o t h e r</code></pre>
<p><em>Test file with words to score</em></p>
</div>
                """)

    # Download and resources section
    with gr.Accordion("📥 Download & Resources", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h3>📦 Sample Data Downloads</h3>
                <p>Access example datasets and templates:</p>
                <p><a href="https://github.com/connormayer/uci_phonotactic_calculator/tree/main/data/sample_test_data" target="_blank">GitHub Sample Data</a></p>
                </div>
                """)
            
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h3>📚 Additional Help</h3>
                <p>Need more detailed format info?</p>
                <p>Check the <strong>About</strong> tab for complete formatting specifications and technical details.</p>
                </div>
                """)

    gr.Markdown("""
    <div class='feature-box'>
    <h2>🆘 Need Help?</h2>
    <ul>
    <li>Try the <strong>Examples</strong> tab for pre-configured demonstrations</li>
    <li>Check the <strong>About</strong> tab for technical details on the models</li>
    <li>Visit our <a href="https://github.com/connormayer/uci_phonotactic_calculator" target="_blank">GitHub repository</a> for more information</li>
    </ul>
    </div>
    """)


# -------------------------------------------------------------------- TAB 3 – GitHub
def _github_tab() -> None:
    gr.Markdown("<div class='feature-box'><strong>🔗 GitHub Repository</strong></div>")


    # Quick links section
    with gr.Accordion("🚀 Quick Links", open=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h3>📚 Documentation</h3>
                <p>Complete usage guide and API reference</p>
                <p><a href="https://github.com/connormayer/uci_phonotactic_calculator#readme" target="_blank">View README</a></p>
                </div>
                
                <div class='step-card'>
                <h3>🐛 Issues</h3>
                <p>Report bugs or request features</p>
                <p><a href="https://github.com/connormayer/uci_phonotactic_calculator/issues" target="_blank">Issue Tracker</a></p>
                </div>
                """)
            
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h3>📊 Sample Data</h3>
                <p>Example datasets and test files</p>
                <p><a href="https://github.com/connormayer/uci_phonotactic_calculator/tree/main/data" target="_blank">Browse Data</a></p>
                </div>
                
                <div class='step-card'>
                <h3>🤝 Contributing</h3>
                <p>Check out the Repo!</p>
                <p><a href="https://github.com/connormayer/uci_phonotactic_calculator" target="_blank">How to Help</a></p>
                </div>
                """)

    # Features section
    with gr.Accordion("✨ Key Features", open=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div class='input-card'>
                <h3>🖥️ Multiple Interfaces</h3>
                <ul>
                <li><strong>Command-line interface:</strong> Run from terminal</li>
                <li><strong>Python API:</strong> Integrate with your code</li>
                <li><strong>Web interface:</strong> This Gradio UI</li>
                </ul>
                </div>
                """)
            
            with gr.Column():
                gr.Markdown("""
                <div class='input-card'>
                <h3>🔧 Advanced Features</h3>
                <ul>
                <li><strong>Plugin system:</strong> Add custom models</li>
                <li><strong>Cross-platform:</strong> Windows, macOS, Linux</li>
                <li><strong>Comprehensive docs:</strong> Detailed explanations</li>
                </ul>
                </div>
                """)

    # Installation section
    with gr.Accordion("💻 Installation", open=True):
        gr.Markdown("""
<div class='input-card'>
<h3>📦 Install from PyPI (Recommended)</h3>
<pre><code>pip install uci-phonotactic-calculator</code></pre>
<p><em>This installs the latest stable version from the Python Package Index.</em></p>
</div>
<div class='input-card'>
<h3>⚡ Install from Source</h3>
<pre><code>git clone https://github.com/connormayer/uci_phonotactic_calculator.git
cd uci_phonotactic_calculator
pip install -e .</code></pre>
<p><em>This installs the development version with latest features.</em></p>
</div>
        """)

    # Contributing section
    with gr.Accordion("🤝 How to Contribute", open=True):
        gr.Markdown("""
        <div class='input-card'>
        <h3>🚀 Ways to Help</h3>
        <p>Contributions are welcome! Here's how you can help:</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h4>🐛 Report Issues</h4>
                <p>Found a bug? Let us know!</p>
                <ul>
                <li>Open detailed bug reports</li>
                <li>Include steps to reproduce</li>
                <li>Share error messages</li>
                </ul>
                </div>
                
                <div class='step-card'>
                <h4>💡 Request Features</h4>
                <p>Have ideas for improvements?</p>
                <ul>
                <li>Suggest new functionality</li>
                <li>Propose UI improvements</li>
                <li>Share use case examples</li>
                </ul>
                </div>
                """)
            
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h4>🔧 Add Plugins</h4>
                <p>Extend the calculator!</p>
                <ul>
                <li>Create new phonotactic models</li>
                <li>Add custom metrics</li>
                <li>Implement new algorithms</li>
                </ul>
                </div>
                
                <div class='step-card'>
                <h4>📝 Improve Docs</h4>
                <p>Help others understand!</p>
                <ul>
                <li>Clarify explanations</li>
                <li>Add examples</li>
                <li>Fix typos and errors</li>
                </ul>
                </div>
                """)


# -------------------------------------------------------------------- TAB 4 – About
def _about_tab() -> None:
    gr.Markdown("<div class='feature-box'><strong>ℹ️ About UCI Phonotactic Calculator</strong></div>")


    # Data format section with better styling
    with gr.Accordion("📄 Data Format Requirements", open=True):
        gr.Markdown("""
        <div class='input-card'>
        <h3>💡 Quick Start</h3>
        <p>The simplest way to understand the format is to look at examples on the <strong>Datasets</strong> tab, or try the demo data!</p>
        </div>
        
        <div class='input-card'>
        <h3>📊 Training File Format</h3>
        <ul>
        <li><strong>Format:</strong> Comma-separated (.csv) file with no headers</li>
        <li><strong>Column 1 (Required):</strong> Word list with symbols separated by spaces<br>
            <em>Example: "cat" becomes "k æ t" in IPA</em></li>
        <li><strong>Column 2 (Optional):</strong> Raw frequency counts</li>
        </ul>
        </div>
        
        <div class='input-card'>
        <h3>🎯 Test File Format</h3>
        <ul>
        <li><strong>Format:</strong> Single column CSV</li>
        <li><strong>Same transcription system</strong> as training file</li>
        <li><strong>Space-separated symbols</strong> (e.g., "d ɔ g" for "dog")</li>
        </ul>
        </div>
        """)

    # Metrics explanation with better organization
    with gr.Accordion("📈 Understanding the Metrics", open=False):
        gr.Markdown("""
        <div class='input-card'>
        <h3>🔢 Core Metrics</h3>
        <p>The calculator provides several types of phonotactic scores:</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h4>📊 Unigram Probabilities</h4>
                <ul>
                <li><code>uni_prob</code> - Basic unigram probability</li>
                <li><code>uni_prob_freq_weighted</code> - Frequency-weighted</li>
                <li><code>uni_prob_smoothed</code> - Add-one smoothed</li>
                <li><code>uni_prob_freq_weighted_smoothed</code> - Both variants</li>
                </ul>
                </div>
                
                <div class='step-card'>
                <h4>🔗 Bigram Probabilities</h4>
                <ul>
                <li><code>bi_prob</code> - Basic bigram probability</li>
                <li><code>bi_prob_freq_weighted</code> - Frequency-weighted</li>
                <li><code>bi_prob_smoothed</code> - Add-one smoothed</li>
                <li><code>bi_prob_freq_weighted_smoothed</code> - Both variants</li>
                </ul>
                </div>
                """)
            
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h4>📍 Positional Unigram Scores</h4>
                <ul>
                <li><code>pos_uni_score</code> - Position-sensitive unigram</li>
                <li><code>pos_uni_score_freq_weighted</code> - Frequency-weighted</li>
                <li><code>pos_uni_score_smoothed</code> - Add-one smoothed</li>
                <li><code>pos_uni_score_freq_weighted_smoothed</code> - Both</li>
                </ul>
                </div>
                
                <div class='step-card'>
                <h4>🎯 Positional Bigram Scores</h4>
                <ul>
                <li><code>pos_bi_score</code> - Position-sensitive bigram</li>
                <li><code>pos_bi_score_freq_weighted</code> - Frequency-weighted</li>
                <li><code>pos_bi_score_smoothed</code> - Add-one smoothed</li>
                <li><code>pos_bi_score_freq_weighted_smoothed</code> - Both</li>
                </ul>
                </div>
                """)

        gr.Markdown("""
        <div class='input-card'>
        <h3>🔬 Metric Variants Explained</h3>
        <ul>
        <li><strong>Frequency-weighted:</strong> High-frequency words contribute more to probability calculations</li>
        <li><strong>Smoothed:</strong> Assigns small probabilities to unseen sound combinations</li>
        <li><strong>Positional:</strong> Considers where sounds appear in words (beginning, middle, end)</li>
        </ul>
        </div>
        """)

    # Technical details
    with gr.Accordion("⚙️ Technical Details", open=False):
        gr.Markdown("""
        <div class='input-card'>
        <h3>🧪 Smoothing Method</h3>
        <p>In the token-weighted versions of the metrics, smoothing is also done by adding one to the log-weighted counts.</p>
        </div>
        
        <div class='input-card'>
        <h3>📚 References</h3>
        <p>Vitevitch, M.S., & Luce, P.A. (2004). A web-based interface to calculate phonotactic probability for words and nonwords in English. 
        <em>Behavior Research Methods, Instruments, & Computers, 36</em>(3), 481-487.</p>
        </div>
        """)

    # Citation and links
    gr.Markdown("""
    <div class='feature-box'>
    <h2>📖 Citation & Links</h2>
    <p><strong>If you use this tool in your research, please cite:</strong></p>
    <blockquote>
    Mayer, C., Kondur, A., & Sundara, M. (2022). UCI Phonotactic Calculator (Version 0.1.0) [Computer software]. 
    <a href="https://doi.org/10.5281/zenodo.7443706" target="_blank">https://doi.org/10.5281/zenodo.7443706</a>
    </blockquote>
    <p>🔗 <a href="https://github.com/connormayer/uci_phonotactic_calculator" target="_blank">GitHub Repository</a> | 
    📧 <a href="https://github.com/connormayer/uci_phonotactic_calculator/issues" target="_blank">Report Issues</a></p>
    </div>
    """)


# -------------------------------------------------------------------- TAB 5 – Documentation
def _docs_tab() -> None:
    gr.Markdown("<div class='feature-box'><strong>📚 How to Use This Interface</strong></div>")

    # Quick Start Guide
    with gr.Accordion("🚀 Quick Start Guide", open=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h3>📁 Step 1: Choose Your Data</h3>
                <ul>
                <li><strong>Upload your own files</strong>: Use CSV files with a 'word' column</li>
                <li><strong>Try the demo</strong>: Click "Load Demo Data" for sample English words</li>
                </ul>
                </div>
                """)
                
                gr.Markdown("""
                <div class='step-card'>
                <h3>⚙️ Step 3: Configure Settings</h3>
                <ul>
                <li><strong>N-gram Order</strong>: How many characters to consider (2-4 recommended)</li>
                <li><strong>Advanced Options</strong>: Fine-tune probability calculations</li>
                </ul>
                </div>
                """)
            
            with gr.Column():
                gr.Markdown("""
                <div class='step-card'>
                <h3>🔧 Step 2: Select a Model</h3>
                <ul>
                <li><strong>N-gram Model</strong>: Classic phonotactic probability</li>
                <li><strong>Positional N-gram</strong>: Position-aware analysis (more sophisticated)</li>
                </ul>
                </div>
                """)
                
                gr.Markdown("""
                <div class='step-card'>
                <h3>📊 Step 4: Calculate & Download</h3>
                <ul>
                <li>Click "Calculate Phonotactic Scores" to process your data</li>
                <li>View results in the preview table</li>
                <li>Download the complete scored dataset as CSV</li>
                </ul>
                </div>
                """)

    # Understanding results section
    with gr.Accordion("📊 Understanding Your Results", open=True):
        gr.Markdown("""
        <div class='input-card'>
        <h3>📈 Score Interpretation</h3>
        <p>The tool adds phonotactic probability scores to each word in your test data:</p>
        <ul>
        <li><strong>Higher scores</strong> = more phonotactically probable (sounds more "natural")</li>
        <li><strong>Lower scores</strong> = less phonotactically probable (sounds more unusual)</li>
        </ul>
        
        <h4>Score Types Available:</h4>
        <ul>
        <li><strong>Raw Probability</strong>: Direct statistical probability</li>
        <li><strong>Log Probability</strong>: Natural logarithm (useful for very small numbers)</li>
        <li><strong>Z-score</strong>: Standardized score comparing to the dataset average</li>
        </ul>
        </div>
        """)

    # Tips section
    with gr.Accordion("💡 Tips for Best Results", open=True):
        gr.Markdown("""
        1. **Training Data**: Use a large, representative sample of your language
        2. **Test Data**: Can be real words, made-up words, or mixed datasets  
        3. **N-gram Order**: Start with 2-3 for most languages, try 4 for detailed analysis
        4. **File Format**: Ensure your CSV has a 'word' column with one word per row
        """)

    # What is phonotactic probability section
    with gr.Accordion("🔬 What is Phonotactic Probability?", open=True):
        gr.Markdown("""
        Phonotactic probability measures how "word-like" a sequence of sounds is in a particular language. It's based on:
        
        - How often sound combinations appear in real words
        - Position-specific patterns (some sounds are more common at word beginnings vs. endings)  
        - Statistical modeling of large word databases
        
        This is useful for linguistics research, psycholinguistics experiments, and understanding sound patterns across languages.
        """)

    # Use cases section  
    with gr.Accordion("📈 Example Use Cases", open=True):
        gr.Markdown("""
        - **Linguistics Research**: Analyze cross-linguistic sound patterns
        - **Psychology Experiments**: Create stimuli with controlled phonotactic properties
        - **Language Learning**: Understand which word combinations sound natural
        - **Computational Linguistics**: Feature engineering for NLP models
        """)

    # Help section
    gr.Markdown("""
    <div class='feature-box'>
    <h2>🆘 Need Help?</h2>
    <ul>
    <li>Try the <strong>Examples</strong> tab for pre-configured demonstrations</li>
    <li>Check the <strong>About</strong> tab for technical details on the models</li>
    <li>Visit our <a href="https://github.com/connormayer/uci_phonotactic_calculator" target="_blank">GitHub repository</a> for more information</li>
    </ul>
    </div>
    """)


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
                _docs_tab()  # Pass None since we're not using components

        _footer()

    return cast(gr.Blocks, demo)


def main() -> None:
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
