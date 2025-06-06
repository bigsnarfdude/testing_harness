"""Language model implementations."""

import asyncio
from typing import Dict, Any, Optional, List
from langchain_ollama import OllamaLLM

from .base import LanguageModel, AsyncLanguageModel
from .config import ModelConfig


class OllamaLanguageModel(LanguageModel):
    """Synchronous Ollama language model implementation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._llm = None
    
    def _get_llm(self):
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = OllamaLLM(
                model=self.config.model_name,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens,
                stop=self.config.stop_sequences,
                **self.config.extra_params
            )
        return self._llm
    
    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt."""
        llm = self._get_llm()
        return llm.invoke(prompt).strip()
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }


class AsyncOllamaLanguageModel(AsyncLanguageModel):
    """Asynchronous Ollama language model implementation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._llm = None
    
    def _get_llm(self):
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = OllamaLLM(
                model=self.config.model_name,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens,
                stop=self.config.stop_sequences,
                **self.config.extra_params
            )
        return self._llm
    
    async def ainvoke(self, prompt: str) -> str:
        """Async invoke the model with a prompt."""
        llm = self._get_llm()
        # Use asyncio.to_thread for non-async libraries
        response = await asyncio.to_thread(llm.invoke, prompt)
        return response.strip()
    
    async def aclose(self):
        """Close any resources."""
        self._llm = None
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }


class MockLanguageModel(LanguageModel):
    """Mock language model for testing."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ["Answer: A"]
        self.call_count = 0
    
    def invoke(self, prompt: str) -> str:
        """Return a mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "provider": "mock",
            "model": "mock-model",
            "responses_count": len(self.responses),
        }


def create_language_model(config: ModelConfig, async_mode: bool = False) -> LanguageModel:
    """Factory function to create language models."""
    if config.provider.lower() == "ollama":
        if async_mode:
            return AsyncOllamaLanguageModel(config)
        else:
            return OllamaLanguageModel(config)
    else:
        raise ValueError(f"Unsupported model provider: {config.provider}")