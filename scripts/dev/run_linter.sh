#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Running Flake8 linter..."
flake8 scripts/
# Add other directories or files if needed

echo "Linting complete."
