"""Tests for configuration management."""

import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path

from testing_harness.config import Config, ModelConfig, load_config, create_default_config


class TestModelConfig:
    """Tests for ModelConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.provider == "ollama"
        assert config.model_name == "llama3.2:latest"
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.0
        assert config.max_tokens == 10
        assert config.stop_sequences == ["\n"]
        assert config.timeout == 30.0
        assert config.extra_params == {}
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            model_name="custom-model",
            temperature=0.5,
            max_tokens=50,
            extra_params={"top_p": 0.9}
        )
        
        assert config.model_name == "custom-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 50
        assert config.extra_params == {"top_p": 0.9}


class TestConfig:
    """Tests for main Config class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        
        assert config.model.provider == "ollama"
        assert config.retry.max_retries == 4
        assert config.runner.batch_size == 5
        assert config.logging.level == "INFO"
        assert config.output_dir == "results"
        assert config.output_format == "json"
        assert config.verbose is False
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "model": {
                "model_name": "test-model",
                "temperature": 0.7
            },
            "retry": {
                "max_retries": 3
            },
            "output_dir": "custom_results",
            "verbose": True
        }
        
        config = Config.from_dict(data)
        
        assert config.model.model_name == "test-model"
        assert config.model.temperature == 0.7
        assert config.retry.max_retries == 3
        assert config.output_dir == "custom_results"
        assert config.verbose is True
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config.model.model_name = "test-model"
        config.verbose = True
        
        data = config.to_dict()
        
        assert data["model"]["model_name"] == "test-model"
        assert data["verbose"] is True
        assert "retry" in data
        assert "runner" in data
        assert "logging" in data
    
    def test_merge_with_env(self, monkeypatch):
        """Test merging configuration with environment variables."""
        # Set environment variables
        monkeypatch.setenv("HARNESS_MODEL", "env-model")
        monkeypatch.setenv("HARNESS_TEMPERATURE", "0.8")
        monkeypatch.setenv("HARNESS_MAX_WORKERS", "8")
        monkeypatch.setenv("HARNESS_VERBOSE", "true")
        
        config = Config()
        config.merge_with_env()
        
        assert config.model.model_name == "env-model"
        assert config.model.temperature == 0.8
        assert config.runner.max_workers == 8
        assert config.verbose is True
    
    def test_save_json(self):
        """Test saving configuration to JSON file."""
        config = Config()
        config.model.model_name = "test-model"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            config.save(config_path)
            
            # Verify file was created and contains correct data
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            assert data["model"]["model_name"] == "test-model"
            
        finally:
            os.unlink(config_path)
    
    def test_save_yaml(self):
        """Test saving configuration to YAML file."""
        config = Config()
        config.model.model_name = "test-model"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            config.save(config_path)
            
            # Verify file was created and contains correct data
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            assert data["model"]["model_name"] == "test-model"
            
        finally:
            os.unlink(config_path)


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_json_config(self):
        """Test loading JSON configuration file."""
        config_data = {
            "model": {"model_name": "json-model"},
            "verbose": True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            assert config.model.model_name == "json-model"
            assert config.verbose is True
            
        finally:
            os.unlink(config_path)
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration file."""
        config_data = {
            "model": {"model_name": "yaml-model"},
            "output_dir": "yaml_results"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            assert config.model.model_name == "yaml-model"
            assert config.output_dir == "yaml_results"
            
        finally:
            os.unlink(config_path)
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_load_default_config(self):
        """Test loading default configuration when no file specified."""
        # Should create default config since no config file exists
        config = load_config(None)
        
        assert isinstance(config, Config)
        assert config.model.provider == "ollama"
    
    def test_unsupported_format(self):
        """Test loading unsupported configuration format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid config")
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported config format"):
                load_config(config_path)
                
        finally:
            os.unlink(config_path)


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""
    
    def test_create_yaml_config(self):
        """Test creating default YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = create_default_config(
                Path(temp_dir) / "test_config", 
                format="yaml"
            )
            
            assert config_path.suffix == ".yaml"
            assert config_path.exists()
            
            # Verify content
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            assert "model" in data
            assert "retry" in data
            assert "runner" in data
    
    def test_create_json_config(self):
        """Test creating default JSON configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = create_default_config(
                Path(temp_dir) / "test_config",
                format="json"
            )
            
            assert config_path.suffix == ".json"
            assert config_path.exists()
            
            # Verify content
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            assert "model" in data
            assert "retry" in data
            assert "runner" in data
    
    def test_unsupported_format(self):
        """Test creating config with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            create_default_config("test_config", format="xml")