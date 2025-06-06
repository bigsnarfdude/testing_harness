"""Pytest configuration and fixtures."""

import pytest
import tempfile
import os
from pathlib import Path

from testing_harness.config import Config, ModelConfig
from testing_harness.models import MockLanguageModel


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    csv_content = """What is 2+2?,2,3,4,5,C
What color is the sky?,Red,Blue,Green,Yellow,B
What is the capital of France?,London,Paris,Berlin,Madrid,B"""
    
    csv_path = temp_dir / "sample_questions.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    
    return csv_path


@pytest.fixture
def gpqa_csv_file(temp_dir):
    """Create a sample GPQA format CSV file for testing."""
    csv_content = """Question,Correct Answer,Incorrect Answer 1,Incorrect Answer 2,Incorrect Answer 3
What is the atomic number of carbon?,6,8,12,14
Which planet is closest to the sun?,Mercury,Venus,Earth,Mars"""
    
    csv_path = temp_dir / "gpqa_sample.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    
    return csv_path


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return Config.from_dict({
        "model": {
            "provider": "ollama",
            "model_name": "test-model",
            "temperature": 0.0
        },
        "retry": {
            "max_retries": 2,
            "retry_delay": 0.1
        },
        "runner": {
            "batch_size": 2,
            "rate_limit_delay": 0.0,
            "checkpoint_interval": 5
        },
        "output_dir": "test_results",
        "verbose": False
    })


@pytest.fixture
def mock_model():
    """Create a mock language model for testing."""
    responses = [
        "Answer: A",
        "Answer: B", 
        "Answer: C",
        "Answer: A"  # Cycling back
    ]
    return MockLanguageModel(responses)


@pytest.fixture
def model_config():
    """Create a test model configuration."""
    return ModelConfig(
        provider="ollama",
        model_name="test-model",
        base_url="http://localhost:11434",
        temperature=0.0,
        max_tokens=10
    )