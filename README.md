# Testing Harness

A unified framework for evaluating Large Language Models (LLMs) on multiple choice benchmarks.

## Features

- **Unified Framework**: Single codebase supporting multiple benchmark formats (MMLU, GPQA, custom CSV)
- **Async Support**: High-performance async evaluation for faster processing
- **Robust Error Handling**: Automatic retry with exponential backoff, checkpoint saving/resuming
- **Flexible Configuration**: YAML/JSON config files with environment variable overrides
- **Multiple Output Formats**: JSON and CSV result export
- **Rich Console Output**: Beautiful progress bars and result tables
- **Comprehensive Logging**: Detailed logging with file rotation

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd testing_harness

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

```bash
# Create a default configuration file
harness config create config.yaml

# Run evaluation on MMLU dataset
harness run benchmark_data/mmlu --config config.yaml

# Run async evaluation on GPQA dataset
harness run benchmark_data/gpqa_diamond.csv --async

# Run with custom model and output directory
harness run data/ --model llama3.2:3b --output-dir results/experiment1
```

## Configuration

### Configuration File

Create a configuration file to customize behavior:

```bash
harness config create my_config.yaml
```

Example configuration:

```yaml
model:
  provider: ollama
  model_name: llama3.2:latest
  base_url: http://localhost:11434
  temperature: 0.0
  max_tokens: 10

retry:
  max_retries: 4
  retry_delay: 2.0
  exponential_backoff: true

runner:
  batch_size: 5
  max_workers: 4
  rate_limit_delay: 1.0
  checkpoint_interval: 10

logging:
  level: INFO
  file_handler: true
  console_handler: true

output_dir: results
output_format: json
verbose: false
```

### Environment Variables

Override configuration with environment variables:

```bash
export HARNESS_MODEL=phi-4
export HARNESS_BASE_URL=http://localhost:11434
export HARNESS_OUTPUT_DIR=results
export HARNESS_VERBOSE=true

harness run benchmark_data/mmlu
```

## Supported Formats

### MMLU Format
Standard CSV with columns: `question, option_a, option_b, option_c, option_d, correct_answer`

### GPQA Format
CSV with columns: `Question, Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3`

### Custom CSV Format
Any CSV following the standard format (question + 4 options + correct answer)

## Data Sources

The harness can process:
- **Single CSV file**: `harness run data/questions.csv`
- **Directory of CSV files**: `harness run benchmark_data/mmlu/`
- **Mixed datasets**: All CSV files in a directory will be processed

## Output

Results are saved in multiple formats:

### JSON Output
```json
{
  "name": "Benchmark Evaluation",
  "test_cases": [...],
  "metadata": {
    "total_questions": 100,
    "passed_tests": 85,
    "accuracy": 85.0,
    "model_config": {...}
  }
}
```

### CSV Output
Tabular format with columns: `id, question, expected, actual, passed, duration, retries, error`

### Summary Report
Human-readable text report with detailed analytics:

```
==================================================
TESTING HARNESS SUMMARY REPORT
==================================================
Test Suite: Benchmark Evaluation
Total Questions: 100
Correct Answers: 85
Accuracy: 85.00%
Average Duration: 2.34s
==================================================
```

## Advanced Features

### Checkpoint and Resume

Long-running evaluations automatically save checkpoints:

```bash
# If interrupted, simply re-run the same command
harness run large_dataset/ --config config.yaml
# Will automatically resume from the last checkpoint
```

### Async Processing

For better performance on large datasets:

```bash
harness run benchmark_data/mmlu --async --config config.yaml
```

### Custom Model Configuration

```bash
# Use different model
harness run data/ --model phi-4 --base-url http://localhost:11434

# With custom parameters via config file
model:
  model_name: custom-model
  temperature: 0.1
  max_tokens: 50
  extra_params:
    top_p: 0.9
    repeat_penalty: 1.1
```

## CLI Reference

### Commands

- `harness run <data_source>` - Run benchmark evaluation
- `harness config create <path>` - Create default configuration
- `harness config show` - Display current configuration

### Run Options

- `--config, -c` - Configuration file path
- `--output-dir, -o` - Output directory
- `--model, -m` - Model name override
- `--async` - Use async runner
- `--format` - Output format (json/csv)
- `--verbose, -v` - Enable verbose logging
- `--resume` - Resume from checkpoint

## Programming Interface

Use the harness programmatically:

```python
from testing_harness import Config, BenchmarkRunner, OllamaLanguageModel

# Load configuration
config = Config.from_dict({
    "model": {"model_name": "llama3.2:latest"},
    "output_dir": "results"
})

# Create model and runner
model = OllamaLanguageModel(config.model)
runner = BenchmarkRunner(model, config)

# Run evaluation
test_suite = runner.run("data/questions.csv")
print(f"Accuracy: {test_suite.get_analytics()['accuracy']:.2f}%")
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=testing_harness tests/
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

1. **Model not responding**: Check if Ollama is running and model is pulled
2. **Permission errors**: Ensure write permissions for output directory
3. **Memory issues**: Reduce batch size in configuration
4. **Network timeouts**: Increase timeout in model configuration

### Debug Mode

Enable verbose logging for debugging:

```bash
harness run data/ --verbose
```

Or set in configuration:
```yaml
logging:
  level: DEBUG
verbose: true
```

## License

[Specify your license here]

## Contributing

[Contribution guidelines here]