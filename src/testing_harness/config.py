"""Configuration management for the testing harness."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """Configuration for language models."""
    provider: str = "ollama"
    model_name: str = "llama3.2:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 10
    stop_sequences: list = field(default_factory=lambda: ["\n"])
    timeout: float = 30.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry policies."""
    max_retries: int = 4
    retry_delay: float = 2.0
    exponential_backoff: bool = True
    backoff_factor: float = 2.0


@dataclass
class RunnerConfig:
    """Configuration for test runners."""
    batch_size: int = 5
    max_workers: int = 4
    rate_limit_delay: float = 1.0
    checkpoint_interval: int = 10
    save_partial_results: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration class for the testing harness."""
    model: ModelConfig = field(default_factory=ModelConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output_dir: str = "results"
    output_format: str = "json"  # json or csv
    verbose: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        config = cls()
        
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "retry" in data:
            config.retry = RetryConfig(**data["retry"])
        if "runner" in data:
            config.runner = RunnerConfig(**data["runner"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        
        # Set top-level attributes
        for key in ["output_dir", "output_format", "verbose"]:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "model": asdict(self.model),
            "retry": asdict(self.retry),
            "runner": asdict(self.runner),
            "logging": asdict(self.logging),
            "output_dir": self.output_dir,
            "output_format": self.output_format,
            "verbose": self.verbose,
        }
    
    def merge_with_env(self):
        """Merge configuration with environment variables."""
        # Model configuration from env
        if env_model := os.getenv("HARNESS_MODEL"):
            self.model.model_name = env_model
        if env_base_url := os.getenv("HARNESS_BASE_URL"):
            self.model.base_url = env_base_url
        if env_temp := os.getenv("HARNESS_TEMPERATURE"):
            self.model.temperature = float(env_temp)
        
        # Runner configuration from env
        if env_workers := os.getenv("HARNESS_MAX_WORKERS"):
            self.runner.max_workers = int(env_workers)
        if env_batch := os.getenv("HARNESS_BATCH_SIZE"):
            self.runner.batch_size = int(env_batch)
        
        # Output configuration from env
        if env_output := os.getenv("HARNESS_OUTPUT_DIR"):
            self.output_dir = env_output
        if env_format := os.getenv("HARNESS_OUTPUT_FORMAT"):
            self.output_format = env_format
        
        # Logging configuration from env
        if env_log_level := os.getenv("HARNESS_LOG_LEVEL"):
            self.logging.level = env_log_level
        if env_verbose := os.getenv("HARNESS_VERBOSE"):
            self.verbose = env_verbose.lower() in ("true", "1", "yes")
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix == ".json":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif path.suffix in (".yaml", ".yml"):
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file or create default."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    if path is None:
        # Look for config files in common locations
        config_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path("config.json"),
            Path(".harness.yaml"),
            Path(".harness.yml"),
            Path(".harness.json"),
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                path = config_path
                break
    
    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            if path.suffix == ".json":
                data = json.load(f)
            elif path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        config = Config.from_dict(data)
    else:
        # Create default config
        config = Config()
    
    # Merge with environment variables
    config.merge_with_env()
    
    return config


def create_default_config(path: Union[str, Path], format: str = "yaml"):
    """Create a default configuration file."""
    config = Config()
    
    path = Path(path)
    if format == "yaml":
        path = path.with_suffix(".yaml")
    elif format == "json":
        path = path.with_suffix(".json")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    config.save(path)
    return path