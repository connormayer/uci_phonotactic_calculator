#!/usr/bin/env python
"""
Main entry point for running the UCI Phonotactic Calculator web UI.
Run with: python -m uci_phonotactic_calculator.web
"""

from uci_phonotactic_calculator.web.gradio.web_demo_v2 import build_ui

if __name__ == "__main__":
    print("Starting web server and opening browser automatically...")
    demo = build_ui()
    demo.queue(max_size=10)
    demo.launch(inbrowser=True, show_error=True)
