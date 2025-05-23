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
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


# --- Progress tracking for the Gradio UI ---
from uci_phonotactic_calculator.utils.progress_base import GradioProgress


# ---> public, documented API wrapper around the CLI
from uci_phonotactic_calculator.cli.demo_data import get_demo_paths
from uci_phonotactic_calculator.cli.legacy import run as ngram_run
from uci_phonotactic_calculator.plugins import PluginRegistry

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
    use_demo,  # bool
    filter_string,  # str like "weight_mode=raw prob_mode=joint"
    hide_progress,  # bool
):
    """
    Execute the scorer and return (DataFrame, CSV-path) for Gradio.

    Parameters:
        train_csv: Training corpus CSV file or None if using demo data
        test_csv: Test corpus CSV file or None if using demo data
        model: Model plugin name to use
        run_full_grid: Whether to run all model variants
        ngram_order: The n-gram order to use (1-4)
        use_demo: Whether to use the packaged demo data
        filter_string: Space-separated key=value pairs for filtering
        hide_progress: Whether to hide the progress indicator

    Returns:
        Tuple of (DataFrame preview, CSV file path)
    """
    # -------------------- resolve input paths -----------------------
    logger.info(
        f"Starting scoring with model={model}, n={ngram_order}, use_demo={use_demo}"
    )

    if use_demo:
        train_path, test_path = get_demo_paths()
    else:
        if train_csv is None or test_csv is None:
            raise gr.Error(
                "Upload BOTH training & test CSVs *or* tick the demo-data box."
            )
        train_path, test_path = train_csv.name, test_csv.name

    # ------------------------------------------------------------------
    # Legacy-mode override for demo data
    # ------------------------------------------------------------------
    if use_demo:
        run_full_grid = False  # ignore any mischievous client-side tweak
        model = None  # guarantees legacy path (no --model)

    out_file = TMP_DIR / f"scores_{uuid.uuid4().hex}.csv"
    atexit.register(functools.partial(out_file.unlink, missing_ok=True))

    # -------------------- translate filters -------------------------
    filters = {}
    tokens = filter_string.split()
    if tokens and tokens[0] == "--filter":
        tokens = tokens[1:]  # drop the flag if present
    if tokens:
        for tok in tokens:
            if "=" not in tok:
                raise gr.Error(f"Filter '{tok}' must look like key=value")
            k, v = tok.split("=", 1)
            filters[k] = v

    # -------------------- invoke library with Gradio progress patch ---------------------------
    from uci_phonotactic_calculator.utils.progress import progress

    _orig_progress = progress  # keep original

    def _gradio_progress(enabled=True):
        return GradioProgress(enabled=enabled and not hide_progress)

    # Patch the progress function
    import uci_phonotactic_calculator.utils.progress as _p

    _p.progress = _gradio_progress
    try:
        ngram_run(
            train_file=train_path,
            test_file=test_path,
            output_file=str(out_file),
            model=None if run_full_grid else model,
            run_all=run_full_grid,
            filters=filters,
            show_progress=not hide_progress,  # still disables library chatter
            extra_args=["-n", str(ngram_order)],
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
def build_ui():
    """
    Build and return the Gradio Blocks interface for the UCI Phonotactic Calculator.
    The interface includes the following tabs:
    - Home/About: Information about the calculator and how to use it
    - Calculator: The main calculator interface for scoring
    - Examples: Pre-configured examples to help users get started
    - Documentation: Detailed documentation on the available options
    """
    with gr.Blocks(title="UCI Phonotactic Calculator", theme=gr.themes.Soft()) as demo:
        # Header section
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                gr.Markdown(
                    "# UCI Phonotactic Calculator\n"
                    "A tool for calculating phonotactic probability scores using n-gram models"
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    "<div style='text-align: right;'>"
                    "<a href='https://github.com/connormayer/uci_phonotactic_calculator' target='_blank'>"
                    "View on GitHub</a></div>"
                )

        # Create tabs for better organization
        with gr.Tabs() as tabs:
            # ===== Home/About Tab =====
            with gr.TabItem("Home"):
                gr.Markdown(
                    "## Welcome to the UCI Phonotactic Calculator\n\n"
                    "This tool helps you calculate phonotactic probability scores "
                    "for phoneme sequences. It is particularly useful for linguistic research, "
                    "including phonology, language acquisition, and psycholinguistics.\n\n"
                    "### What does it do?\n\n"
                    "The calculator takes in a training corpus and a test corpus, both in CSV format, "
                    "and calculates probability scores for the test items based on n-gram statistics "
                    "derived from the training corpus.\n\n"
                    "### How to use it\n\n"
                    "1. Go to the **Calculator** tab\n"
                    "2. Either upload your own training and test corpora, or use the built-in English demo data\n"
                    "3. Choose a model and settings\n"
                    "4. Click 'Score' to run the calculator\n"
                    "5. View the results and download the full CSV\n\n"
                    "### Need help?\n\n"
                    "Check out the **Examples** tab for pre-configured examples, or the **Documentation** tab "
                    "for detailed information about the available options."
                )

            # ===== Calculator Tab =====
            with gr.TabItem("Calculator"):
                with gr.Row():
                    # Input column
                    with gr.Column():
                        gr.Markdown("### Input")
                        with gr.Group():
                            gr.Markdown("#### Data Selection")
                            use_demo = gr.Checkbox(
                                label="Use packaged English demo data", value=True
                            )
                            gr.Markdown(
                                "*Uses the built-in English demo data for training and testing*"
                            )
                            # Custom data inputs row will be toggled based on use_demo value
                            with gr.Row(visible=False) as custom_data_row:
                                with gr.Column():
                                    train_in = gr.File(
                                        label="Training CSV", file_types=["csv"]
                                    )
                                    gr.Markdown("*CSV file containing training corpus*")

                                with gr.Column():
                                    test_in = gr.File(
                                        label="Test CSV", file_types=["csv"]
                                    )
                                    gr.Markdown("*CSV file containing test items*")

                        with gr.Group():
                            gr.Markdown("#### Model Configuration")
                            with gr.Row():
                                model_dd = gr.Dropdown(
                                    choices=sorted(PluginRegistry),
                                    value="ngram",
                                    label="Model",
                                )
                                gr.Markdown("*Select the model to use for scoring*")

                                run_grid = gr.Checkbox(
                                    label="Run full variant grid", value=False
                                )
                                gr.Markdown("*Run all variants of the selected model*")

                            n_slider = gr.Slider(
                                minimum=1,
                                maximum=4,
                                step=1,
                                value=2,
                                label="n-gram order",
                            )
                            gr.Markdown(
                                "*The order of n-grams to use*"
                                "*(1 = unigrams, 2 = bigrams, etc.)*"
                            )

                        with gr.Accordion("Advanced Options", open=False):
                            filt_txt = gr.Textbox(
                                label="Filters",
                                placeholder="example: weight_mode=raw prob_mode=joint",
                            )
                            gr.Markdown(
                                "*Space-separated key=value pairs for*"
                                "*filtering results*"
                            )

                            hide_prog = gr.Checkbox(
                                label="Hide progress indicator", value=False
                            )
                            gr.Markdown("*Hide the progress bar during calculation*")

                        go_btn = gr.Button("Score", variant="primary")

                    # Output column
                    with gr.Column():
                        gr.Markdown("### Results")
                        out_df = gr.Dataframe(label="Score Preview", interactive=False)
                        out_csv = gr.File(label="Download Full Results (CSV)")

                # Define function to toggle visibility of custom data inputs
                def toggle_custom_data(use_demo_val):
                    return gr.Row(visible=not use_demo_val)

                # Make the custom data row visible only when use_demo is False
                use_demo.change(
                    fn=toggle_custom_data, inputs=[use_demo], outputs=[custom_data_row]
                )

                # Connect the score button to the score function
                go_btn.click(
                    fn=score,
                    inputs=[
                        train_in,
                        test_in,
                        model_dd,
                        run_grid,
                        n_slider,
                        use_demo,
                        filt_txt,
                        hide_prog,
                    ],
                    outputs=[out_df, out_csv],
                )

            # ===== Examples Tab =====
            with gr.TabItem("Examples"):
                gr.Markdown(
                    "## Examples\n\n"
                    "Below are some pre-configured examples to help you get started."
                    "Click on any example to load it into the calculator."
                )

                with gr.Accordion("Example 1: English Demo with Bigrams", open=True):
                    gr.Markdown(
                        "This example uses the built-in English demo data with "
                        "bigram (n=2) scoring. It's a good starting point to "
                        "understand how the calculator works."
                    )
                    example1_btn = gr.Button("Load this example")

                with gr.Accordion("Example 2: English Demo with Trigrams", open=False):
                    gr.Markdown(
                        "This example uses the built-in English demo data with "
                        "trigram (n=3) scoring. Higher n-gram orders can capture "
                        "more context but may require more training data."
                    )
                    example2_btn = gr.Button("Load this example")

                with gr.Accordion("Example 3: Raw Joint Probability", open=False):
                    gr.Markdown(
                        "This example uses the built-in English demo data with "
                        "specific filters for raw joint probability calculation."
                    )
                    example3_btn = gr.Button("Load this example")

                # Example button handlers
                def load_example1():
                    return True, "ngram", False, 2, "", False

                def load_example2():
                    return True, "ngram", False, 3, "", False

                def load_example3():
                    return (
                        True,
                        "ngram",
                        False,
                        2,
                        "weight_mode=raw prob_mode=joint",
                        False,
                    )

                # Connect example buttons to handlers
                example1_btn.click(
                    fn=load_example1,
                    inputs=[],
                    outputs=[
                        use_demo,
                        model_dd,
                        run_grid,
                        n_slider,
                        filt_txt,
                        hide_prog,
                    ],
                )

                example2_btn.click(
                    fn=load_example2,
                    inputs=[],
                    outputs=[
                        use_demo,
                        model_dd,
                        run_grid,
                        n_slider,
                        filt_txt,
                        hide_prog,
                    ],
                )

                example3_btn.click(
                    fn=load_example3,
                    inputs=[],
                    outputs=[
                        use_demo,
                        model_dd,
                        run_grid,
                        n_slider,
                        filt_txt,
                        hide_prog,
                    ],
                )

            # ===== Documentation Tab =====
            with gr.TabItem("Documentation"):
                gr.Markdown(
                    "## Documentation\n\n"
                    "### Input Files\n\n"
                    "The calculator expects two CSV files:\n\n"
                    "1. **Training Corpus**: A CSV file containing the training corpus\n"
                    "   from which n-gram statistics will be calculated.\n"
                    "2. **Test Corpus**: A CSV file containing the test items\n"
                    "   for which probability scores will be calculated.\n\n"
                    "### Data Format\n\n"
                    "The CSV files should follow this format:\n\n"
                    "- Each row represents one word/item\n"
                    "- The first column contains the orthographic form (optional)\n"
                    "- The second column contains the phonemic transcription\n"
                    "- Subsequent columns can contain additional information\n"
                    "  like frequency, part of speech, etc.\n\n"
                    "### Available Models\n\n"
                    "- **ngram**: Standard n-gram model (default)\n"
                    "- Other models as available in the plugins registry\n\n"
                    "### Filter Options\n\n"
                    "Filters allow you to customize the calculation. \n"
                    "Some common filters include:\n\n"
                    "- `weight_mode=raw|log`: Use raw or log probabilities\n"
                    "- `prob_mode=joint|cond`: Use joint or conditional probabilities\n"
                    "- `smoothing=laplace|add_k`: Choose smoothing method\n\n"
                    "For more detailed documentation, please refer to the "
                    "[GitHub repository](https://github.com/connormayer/uci_phonotactic_calculator)."
                )

        # We don't need to set a default tab here since Gradio will
        # use the first tab by default

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
