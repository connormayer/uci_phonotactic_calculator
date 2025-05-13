#!/bin/bash
# release.sh - Automate PyPI release for uci-phonotactic-calculator
# Usage: bash release.sh
set -e

# Remove previous build artifacts
echo "Removing previous build artifacts..."
rm -rf dist build
find . -type d -name "*.egg-info" -exec rm -rf {} +

# Build the package
echo "Building package..."
python -m build

# Upload to PyPI
echo "Uploading package to PyPI..."
python -m twine upload dist/*

echo "Release process completed successfully."
