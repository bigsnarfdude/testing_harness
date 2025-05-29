#!/bin/bash
set -e

echo "Running pip-audit for known vulnerabilities in dependencies..."
# Ensure dependencies from requirements.txt are available for audit
# If running in a CI pipeline, the environment should already have these.
# For local runs, it's assumed the user has installed dependencies
# in their active virtual environment.

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found. Please ensure the script is run from the project root."
    exit 1
fi

pip-audit -r requirements.txt

# Optionally, you could also audit dev dependencies, but it's less critical:
# echo "Running pip-audit for dev dependencies..."
# if [ -f requirements-dev.txt ]; then
#   pip-audit -r requirements-dev.txt
# else
#   echo "Warning: requirements-dev.txt not found. Skipping audit of dev dependencies."
# fi

echo "Security audit complete."
