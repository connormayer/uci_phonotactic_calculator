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
#df-wrapper {
    max-height: 350px;
    overflow: hidden;
}

/* Make the dataframe fill the available space */
#df-wrapper .svelte-16r4veq {
    width: 100%;
}

.download-row {
    display: flex;
    align-items: center;
    gap: 10px;
    background-color: #f8f9fa;
    padding: 10px 15px;
    border-radius: 8px;
    margin-bottom: 15px;
    border-left: 4px solid #2ecc71;
}

.download-row p {
    margin: 0;
    font-size: 0.9rem;
}

.container        { margin:0 auto; }
.feature-box      { background:#f0f7ff; padding:16px; border-radius:8px; margin-bottom:16px; border-left:4px solid #3498db; }
.info-box         { background:#e8f6f3; padding:16px; border-radius:8px; margin-bottom:16px; border-left:4px solid #2ecc71; }
.warning-box      { background:#fff9e6; padding:16px; border-radius:8px; margin-bottom:16px; border-left:4px solid #f39c12; }
.examples-row     { margin-top:8px; }
footer            { margin-top:20px; text-align:center; font-size:.8em; color:#666; }

/* Typography enhancements */
h1, h2, h3, h4 { font-weight:700; color:#2c3e50; }
h1              { font-size:1.8em; margin-top:0.5em; }
h2              { font-size:1.5em; margin-top:1em; }
h3              { font-size:1.2em; margin-top:1em; }
h4              { font-size:1.1em; margin-top:1em; }

/* Color/label highlighting */
.highlight      { background:#ffffcc; padding:2px 4px; border-radius:4px; font-weight:bold; }

/* Better table formatting */
table { width:100%; border-collapse:collapse; margin:16px 0; }
table th { background:#f0f7ff; text-align:left; padding:8px; }
table td { border-top:1px solid #ddd; padding:8px; }
table tr:nth-child(even) { background:#f9f9f9; }

/* New calculator UI enhancements */
.calculator-container { background:#f7f9fc; border-radius:10px; padding:15px; }
.input-panel { background:white; border-radius:8px; border:1px solid #e0e5eb; padding:12px; margin-bottom:15px; }
.option-label { font-weight:bold; color:#2a6ea7; margin-bottom:5px; }
.option-divider { text-align:center; margin:10px 0; color:#7f8c8d; }
.calc-button { background:#3498db; color:white; padding:10px 15px; border-radius:5px; font-weight:bold; cursor:pointer; }
.calc-button:hover { background:#2980b9; }
.results-box { background:#f0f7ff; border-radius:8px; border:1px solid #d0e1f9; padding:15px; }
.training-options-box { border:1px solid #e0e5eb; border-radius:8px; padding:12px 15px; background:#f8f9fa; }

/* Highlight important terms */
.highlight { color:#e74c3c; font-weight:bold; }
.key-term { color:#3498db; font-weight:bold; }

/* Table styling */
table { border-collapse:collapse; width:100%; margin-bottom:1em; }
table, th, td { border:1px solid #ddd; }
th, td { padding:8px 12px; text-align:left; }
th { background-color:#f2f2f2; }
tr:nth-child(even) { background-color:#f9f9f9; }
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

        # For test file, use the uploaded file
        test_path = test_csv.name if not isinstance(test_csv, str) else test_csv
        logger.info(f"Using test file: {test_path}")
    else:
        # Both custom files
        if train_csv is None or test_csv is None:
            raise gr.Error(
                "Upload BOTH training & test CSVs *or* select a default training file."
            )
        train_path, test_path = train_csv.name, test_csv.name
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
                extra_args.extend(["--neighbourhood-mode", v])  # Pass as an actual CLI flag
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
    # Dictionary to store UI components for sharing with other tabs
    components = {}

    gr.Markdown("<div class='calculator-container'>")

    # Title in a compact way
    gr.Markdown(
        "<h3 style='margin:0 0 15px 0;text-align:center;'>UCI Phonotactic Calculator</h3>"
    )

    with gr.Row():
        # Left panel for inputs only
        with gr.Column(scale=1):
            # Training file section with clear either/or choice
            gr.Markdown("<div class='input-panel'>")
            gr.Markdown(
                "<h4 style='margin:0 0 10px 0;color:#2c3e50;'>Training File</h4>"
            )

            with gr.Group(elem_classes="training-options-box"):
                gr.Markdown(
                    "<p class='option-label'>OPTION 1: Select a default file</p>"
                )
                # Default training file selection - dropdown with descriptions
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
                components["default_file"] = gr.Dropdown(
                    label="",
                    choices=[value for _, value in file_choices],
                    interactive=True,
                )

                gr.Markdown("<div class='option-divider'>--- OR ---</div>")

                gr.Markdown(
                    "<p class='option-label'>OPTION 2: Upload your own file</p>"
                )
                components["train_in"] = gr.File(
                    label="",
                    file_types=[".csv"],
                    interactive=True,
                )

            # Demo pair option - gets both train and test files from demo data
            components["use_demo_pair"] = gr.Checkbox(
                label="Use English demo pair (both train & test files)",
                value=False,
                info="Check this to use the default English training and test files. Overrides other file selections.",
            )

            # Test file section (always required)
            gr.Markdown(
                "<h4 style='margin:15px 0 10px 0;color:#2c3e50;'>Test File (required)</h4>"
            )
            components["test_in"] = gr.File(
                label="",
                file_types=[".csv"],
                interactive=True,
            )
            gr.Markdown("</div>")

            # Main submit button
            components["calc_btn"] = gr.Button(
                "Calculate", variant="primary", elem_classes="calc-button"
            )

            # Results display immediately below the Calculate button
            components["results_container"] = gr.Group(visible=False)
            with components["results_container"]:
                gr.Markdown("<div class='results-box'>")
                gr.Markdown(
                    "<h3 style='margin-top:0;color:#2c3e50;'>Calculation Complete!</h3>"
                )

                # Download button with filename display
                with gr.Row(elem_classes=["download-row"]):
                    components["dl_label"] = gr.Markdown(elem_classes=["file-label"])
                    components["out_csv"] = gr.DownloadButton(
                        label="Download CSV", variant="primary", size="sm"
                    )

                # Results table in an accordion that's collapsed by default
                with gr.Accordion("View Results Table", open=False):
                    with gr.Column(elem_id="df-wrapper"):
                        components["out_df"] = gr.Dataframe(label="", interactive=False)

                gr.Markdown("</div>")

        # Right panel for filters, model settings, and options
        with gr.Column(scale=1):
            # Filter options panel
            gr.Markdown("<div class='input-panel'>")
            gr.Markdown(
                "<h4 style='margin:0 0 10px 0;color:#2c3e50;'>Filter Options</h4>"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("<p class='option-label'>Weight Mode</p>")
                    components["weight_mode"] = gr.Radio(
                        ["None", "Raw", "Log"], value="None", label=""
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("<p class='option-label'>Probability</p>")
                    components["prob_mode"] = gr.Radio(
                        ["None", "Joint", "Conditional"], value="None", label=""
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("<p class='option-label'>Smoothing</p>")
                    components["smooth_mode"] = gr.Radio(
                        ["None", "Laplace", "Add-k"], value="None", label=""
                    )

            # Model settings moved to right column
            gr.Markdown("<div class='input-panel'>")
            gr.Markdown(
                "<h4 style='margin:0 0 10px 0;color:#2c3e50;'>Model Settings</h4>"
            )
            # Model dropdown and neighbourhood mode dropdown side by side
            with gr.Row():
                components["model_dd"] = gr.Dropdown(
                    choices=sorted(PluginRegistry), value="ngram", label="Model"
                )

                # Add the neighbourhood mode dropdown (only visible when neighbourhood model is selected)
                components["nh_mode"] = gr.Dropdown(
                    ["full", "substitution_only"],
                    value="full",
                    label="Neighbourhood mode",
                    visible=False,  # Hidden by default
                )

            # N-gram order slider
            components["n_slider"] = gr.Slider(1, 4, 2, step=1, label="N-gram order")
            
            # Add Run full variant grid checkbox
            components["run_grid"] = gr.Checkbox(False, label="Run full variant grid")

            # Add change handlers for model selection
            def update_ui_for_model(model_name):
                # Show neighbourhood mode dropdown only for neighbourhood model
                # Hide Run full variant grid checkbox for neighbourhood model (it has no variants)
                return (
                    gr.update(
                        visible=(model_name == "neighbourhood")
                    ),  # nh_mode visibility
                    gr.update(
                        visible=(model_name != "neighbourhood")
                    ),  # run_grid visibility
                )

            components["model_dd"].change(
                update_ui_for_model,
                inputs=components["model_dd"],
                outputs=[
                    components["nh_mode"],  # Show/hide neighbourhood mode dropdown
                    components["run_grid"],  # Show/hide run full variant grid checkbox
                ],
            )
            gr.Markdown("</div>")  # Close the model settings panel
            gr.Markdown("</div>")

            # Advanced options
            gr.Markdown("<div class='input-panel'>")
            gr.Markdown(
                "<h4 style='margin:0 0 10px 0;color:#2c3e50;'>Advanced Options</h4>"
            )
            components["hide_prog"] = gr.Checkbox(False, label="Hide progress bar")
            components["filt_txt"] = gr.Textbox(
                label="Custom filters",
                placeholder="key=value pairs (e.g., weight_mode=raw prob_mode=joint)",
            )
            gr.Markdown("</div>")

    # No results area at the bottom anymore - moved directly below Calculate button

    gr.Markdown("</div>")  # Close calculator-container

    # ---------- WIRING ---------- #

    def build_filter_string(*args):
        """Build filter string from radio button selections"""
        (
            weight_mode,
            prob_mode,
            smooth_mode,
            neighbourhood_mode,
            model_name,
            advanced_filters,
        ) = args

        # Ignore neighbourhood_mode if not using the neighbourhood model
        if model_name != "neighbourhood":
            neighbourhood_mode = None

        tokens = []

        # Log the selections to help with debugging
        logger.info(
            f"Selected options - Weight: {weight_mode}, Probability: {prob_mode}, Smoothing: {smooth_mode}, Neighbourhood: {neighbourhood_mode}"
        )

        # ─ weight ────────────────────────────
        if weight_mode == "Raw":
            tokens.append("weight_mode=raw")
        elif weight_mode == "Log":
            tokens.append("weight_mode=log")

        # ─ probability ───────────────────────
        # "Joint" stays a filter; "Conditional" becomes a CLI flag later.
        if prob_mode == "Joint":
            tokens.append("prob_mode=joint")

        # ─ smoothing ─────────────────────────
        if smooth_mode == "Laplace":
            tokens.append("smoothing=laplace")
        elif smooth_mode == "Add-k":  # our placeholder smoother
            tokens.append("smoothing_scheme=kneser_ney")

        # ─ neighbourhood ─────────────────────
        # We now pass neighbourhood mode as a CLI flag, not as a filter
        # So we don't need to add it to the filter tokens
        
        # Add the neighborhood mode to extra_args when needed
        extra_args = []
        if model_name == "neighbourhood" and neighbourhood_mode not in (None, "full", "None"):
            extra_args.extend(["--neighbourhood-mode", neighbourhood_mode])

        # ─ anything typed in free-form box ───
        if advanced_filters.strip():
            tokens.append(advanced_filters.strip())

        filter_string = " ".join(tokens)
        logger.info(f"Generated filter string: {filter_string}")
        logger.info(f"Generated extra args for neighbourhood: {extra_args}")
        return filter_string, extra_args

    # We don't need the mutual exclusivity logic for radio buttons since they handle it automatically

    # We no longer need the toggle_demo function since we're using a different UI approach

    # Build filter string and call score function
    def process_and_score(*args):
        (
            default_file,
            train_in,
            use_demo_pair,
            test_in,
            model_dd,
            n_slider,
            run_grid,
            hide_prog,
            weight_mode,
            prob_mode,
            smooth_mode,
            nh_mode,
            filt_txt,
        ) = args

        # Check if the demo pair checkbox is checked
        if use_demo_pair:
            # Get both paths from demo data
            train_path, test_path = get_demo_paths()
            train_in = train_path
            test_in = test_path
            logger.info(f"Using demo pair: {train_path}, {test_path}")
        else:
            # Check if we're using a default file or a custom upload
            using_default_file = isinstance(
                default_file, str
            ) and default_file.endswith(".csv")

            # Validate input - either default OR custom training file must be provided
            if not using_default_file and (train_in is None or train_in == ""):
                return (
                    gr.update(visible=False),
                    None,
                    gr.update(visible=False),
                )

            # Test file is required
            if test_in is None or test_in == "":
                return (
                    gr.update(visible=False),
                    None,
                    gr.update(visible=False),
                )

            # If using a default file, just pass the string name of the default file
            # Resolution to actual file path happens in score()
            if using_default_file:
                train_in = default_file
                logger.info(f"Selected default file: {train_in}")

        filter_string, neighbourhood_extra_args = build_filter_string(
            weight_mode,
            prob_mode,
            smooth_mode,
            nh_mode,
            model_dd,  # Pass the model name to check if it's neighbourhood
            filt_txt,
        )

        # Run the score calculation
        result_df, result_csv = score(
            train_in,
            test_in,
            model_dd,
            run_grid,
            n_slider,
            filter_string,
            hide_prog,
            neighbourhood_extra_args,  # Pass the neighborhood mode extra args
        )

        # Make results container visible after calculation completes
        from pathlib import Path

        return (
            result_df,
            gr.update(value=result_csv),
            gr.update(visible=True),
            gr.update(value=f"**File:** `{Path(result_csv).name}`"),
        )

    components["calc_btn"].click(
        process_and_score,
        [
            components["default_file"],  # Default file selection
            components["train_in"],  # Custom training file upload
            components["use_demo_pair"],  # Use English demo pair checkbox
            components["test_in"],  # Test file (required)
            components["model_dd"],  # Model selection
            components["n_slider"],  # N-gram order
            components["run_grid"],  # Run full grid
            components["hide_prog"],  # Hide progress bar
            components["weight_mode"],  # Weight mode radio
            components["prob_mode"],  # Probability mode radio
            components["smooth_mode"],  # Smoothing mode radio
            components["nh_mode"],  # Neighbourhood mode radio
            components["filt_txt"],  # Custom filters text
        ],
        [
            components["out_df"],
            components["out_csv"],
            components["results_container"],
            components["dl_label"],
        ],
    )

    return components


# -------------------------------------------------------------------- TAB 2 – Datasets
def _datasets_tab(components=None):
    gr.Markdown("<div class='feature-box'><strong>Available Datasets</strong></div>")

    with gr.Accordion("Built-in Demo Datasets", open=True):
        gr.Markdown(
            "The calculator comes with several datasets for demonstration purposes:\n\n"
            "### English Datasets\n"
            "- **english.csv**: Basic English word list\n"
            "- **english_freq.csv**: English words with frequency counts\n"
            "- **english_needle.csv**: English test words\n"
            "- **english_onsets.csv**: English words organized by onset\n\n"
            "### Other Languages\n"
            "- **finnish.csv**: Finnish word list\n"
            "- **french.csv**: French word list\n"
            "- **polish_onsets.csv**: Polish words organized by onset\n"
            "- **samoan.csv**: Samoan word list\n"
            "- **spanish_stress.csv**: Spanish words with stress patterns\n"
            "- **turkish.csv**: Turkish word list\n\n"
            "To use the built-in demo datasets, simply check the 'Use built-in demo data' checkbox in the Calculator tab."
        )

    with gr.Accordion("Using Your Own Data", open=True):
        gr.Markdown(
            "To use your own data, uncheck the 'Use built-in demo data' checkbox in the Calculator tab and upload your own CSV files.\n\n"
            "#### Required format\n"
            "- Both the training and test files must be in CSV format (.csv)\n"
            "- The training file should consist of one or two columns with no headers\n"
            "  * First column (mandatory): Word list with space-separated symbols\n"
            "  * Second column (optional): Word frequencies as raw counts\n"
            "- The test file should consist of a single column containing the test word list\n"
            "- The output file will contain the test words, word length, and all calculated metrics\n\n"
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
        _spacer()

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
