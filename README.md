# LLM Benchmarking Harness

## Project Overview

This project provides an asynchronous benchmarking harness designed to evaluate the performance of Language Learning Models (LLMs) on multiple-choice question datasets. It interacts with an Ollama backend for model serving and provides features for configuration, structured logging, and metrics exposition.

## Features

*   **Configurable Ollama Backend:** Set the target Ollama instance and model.
*   **Asynchronous Processing:** Uses `asyncio` for non-blocking benchmark execution.
*   **Multiple Choice Evaluation:** Supports CSV files for multiple-choice questions.
*   **Structured JSON Logging:** Outputs logs in JSON format.
*   **Metrics Endpoint:** Exposes benchmark analytics (accuracy, etc.) via an HTTP `/metrics` endpoint.
*   **Pydantic Configuration:** Robust configuration via environment variables and CLI arguments.
*   **Docker Support:** `Dockerfile` for containerized execution.
*   **Developer Tools:** Scripts for linting, testing, and security auditing.

## Project Structure

```
.
├── Dockerfile                  # For Docker image
├── README.md                   # This file
├── benchmark_data/             # Placeholder for CSV datasets
├── pyproject.toml              # Project configuration (Flake8, Pytest)
├── requirements-dev.txt        # Development dependencies
├── requirements.txt            # Core application dependencies
├── results/                    # Default output directory (gitignored)
├── scripts/
│   ├── async_harness.py        # Main application script
│   └── dev/                    # Developer scripts (lint, test, audit)
└── tests/
    └── unit/                   # Unit tests
```

## Prerequisites

*   Python 3.9+
*   Docker
*   An accessible Ollama instance (this is a separate service that the harness connects to).

## Configuration

The harness is configured through environment variables (defined in `AppConfig` in `scripts/async_harness.py`), which can be overridden by command-line arguments.

### Key Environment Variables (Defaults shown):

*   `OLLAMA_BASE_URL`: URL of the Ollama API (`http://localhost:11434`).
*   `MODEL_NAME`: Name of the Ollama model to use (`llama3.2:latest`).
*   `OUTPUT_DIR`: Directory to save logs and results (`results`).
*   `VERBOSE`: Set to `true` for verbose DEBUG level logging (`false`).
*   `METRICS_PORT`: Port for the metrics HTTP server (`8000`).
*   `METRICS_HOST`: Host for the metrics HTTP server (`0.0.0.0`).
*   `MAX_RETRIES`: Max retries for a question (`4`).
*   `RATE_LIMIT_DELAY`: Delay between question batches (`1.0`s).
*   `BATCH_SIZE`: Number of questions per batch (`5`).
*   `CSV_PATTERN`: Glob pattern for CSV files (`*.csv`).
*   *For a full list, see `AppConfig` in `scripts/async_harness.py`.*

### Command-Line Arguments (`scripts/async_harness.py`):

These override environment variables and defaults.
*   `csv_dir`: (Positional) Directory containing CSV question files.
*   `--output-dir`: Directory to save results.
*   `--verbose`: Enable verbose logging.
*   `--model`: Model to use for evaluation.
*   `--base-url`: Base URL for the Ollama API.
*   `--max-retries`, `--batch-size`, `--metrics-port`, `--metrics-host`, etc.

Run `python scripts/async_harness.py --help` for all options.

## Running the Harness

### Local Python Execution

1.  **Install Dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    python -m pip install -r requirements.txt
    ```

2.  **Run the Harness:**
    Ensure your Ollama service is running and accessible.
    ```bash
    python scripts/async_harness.py ./benchmark_data/
    ```
    Replace `./benchmark_data/` with the actual path to your directory containing CSV question files.

    **Example with options:**
    ```bash
    python scripts/async_harness.py ./benchmark_data/ \
      --model="phi3:latest" \
      --base-url="http://192.168.1.100:11434" \
      --output-dir="my_custom_results" \
      --verbose
    ```

### Docker Execution

1.  **Build the Docker Image:**
    ```bash
    docker build -t benchmark-harness .
    ```

2.  **Run the Harness using Docker:**
    Ensure your Ollama service is accessible from the Docker container (e.g., use your host IP or a networked Ollama container).

    *   **If `benchmark_data` is copied into the image (as per current Dockerfile):**
        The `benchmark_data/` directory from the build context is copied into `/app/benchmark_data/` in the image.
        ```bash
        docker run -it --rm \
          -e OLLAMA_BASE_URL="http://<ollama_host_or_ip>:11434" \
          -e MODEL_NAME="your_model:latest" \
          -e VERBOSE="true" \
          -p 8000:8000 \ # Expose metrics port
          -v ./results:/app/results \ # Mount local results directory
          benchmark-harness benchmark_data # Path to CSVs inside the container
        ```
        Replace `<ollama_host_or_ip>` with the actual IP or hostname of your Ollama service. If Ollama is running on the same host as Docker, you might use `host.docker.internal` for `OLLAMA_BASE_URL` (platform dependent).

    *   **If providing `benchmark_data` dynamically (e.g., for different datasets without rebuilding):**
        Mount your local data directory to a path inside the container (e.g., `/app/custom_data`) and pass that path as the `csv_dir` argument to the harness.
        ```bash
        docker run -it --rm \
          -e OLLAMA_BASE_URL="http://<ollama_host_or_ip>:11434" \
          -e MODEL_NAME="your_model:latest" \
          -p 8000:8000 \
          -v ./my_local_benchmark_data:/app/custom_data \ # Mount your local data
          -v ./results:/app/results \
          benchmark-harness /app/custom_data # Path inside container
        ```

## Development Setup

1.  **Install Development Dependencies:**
    (Assumes you have already created and activated a virtual environment as shown in Local Python Execution)
    ```bash
    python -m pip install -r requirements-dev.txt
    ```

2.  **Running Linters:**
    Uses Flake8. Configuration is in `pyproject.toml`.
    ```bash
    sh scripts/dev/run_linter.sh
    ```

3.  **Running Tests:**
    Uses Pytest. Configuration is in `pyproject.toml`.
    ```bash
    sh scripts/dev/run_tests.sh
    ```

4.  **Running Security Audits:**
    Uses pip-audit to check for known vulnerabilities in dependencies.
    ```bash
    sh scripts/dev/run_security_audit.sh
    ```

## Output

*   **Logs:** Structured JSON logs are written to a timestamped file (e.g., `test_run_YYYYMMDD_HHMMSS.log`) inside the configured `OUTPUT_DIR` (default: `results/`).
*   **Console Output:** A summary table of results is printed to the console upon completion of the benchmark run.
*   **Metrics Endpoint:** Detailed metrics from the latest run are available via HTTP (see below).

## Metrics Endpoint

The harness exposes a `/metrics` endpoint (default: `http://localhost:8000/metrics` or `http://<container_ip>:8000/metrics` if running in Docker).
This endpoint provides a JSON response with analytics from the latest completed benchmark run, including:
*   Total tests
*   Passed tests
*   Failed tests
*   Accuracy (%)
*   Average duration per question
*   Total retries
*   Error rate (%)

If accessed before any benchmarks are run or while a run is in progress and metrics haven't been calculated yet, it will return `{"status": "no metrics available yet"}`.

## Troubleshooting

*   **Ollama Connection Issues:**
    *   Ensure your Ollama service is running and accessible from where you are running the harness (local machine or Docker container).
    *   Verify `OLLAMA_BASE_URL` is correctly set.
    *   If using Docker, ensure the container can reach the Ollama URL (e.g., use `host.docker.internal` for the host machine on Docker Desktop, or a specific IP address on other systems).
*   **Model Not Found:** Ensure the `MODEL_NAME` specified exists in your Ollama service. You might need to run `ollama pull <model_name>` in your Ollama service.
*   **CSV Parsing Errors:**
    *   Check that your CSV files are in the expected format (question, 4 options, correct letter: A, B, C, or D).
    *   The harness logs warnings for rows with invalid correct answer letters and skips them.
*   **Permission Denied (Docker Volume Mounts):** Ensure Docker has the necessary permissions to read/write to the directories you are mounting for results (e.g., `./results`). This is less common for output directories but can occur.
*   **Python Version:** Ensure you are using Python 3.9 or newer.
*   **Dependencies:** If you encounter import errors, ensure all dependencies are installed by running `pip install -r requirements.txt` (and `requirements-dev.txt` for development).