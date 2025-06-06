"""Tests for language model implementations."""

import pytest
from unittest.mock import Mock, patch

from testing_harness.models import OllamaLanguageModel, MockLanguageModel, create_language_model
from testing_harness.config import ModelConfig


class TestMockLanguageModel:
    """Tests for MockLanguageModel class."""
    
    def test_default_responses(self):
        """Test mock model with default responses."""
        model = MockLanguageModel()
        
        response = model.invoke("Test prompt")
        assert response == "Answer: A"
        
        # Test call counting
        assert model.call_count == 1
        
        # Call again
        response2 = model.invoke("Another prompt")
        assert response2 == "Answer: A"
        assert model.call_count == 2
    
    def test_custom_responses(self):
        """Test mock model with custom responses."""
        responses = ["Answer: B", "Answer: C", "Answer: D"]
        model = MockLanguageModel(responses)
        
        assert model.invoke("Test 1") == "Answer: B"
        assert model.invoke("Test 2") == "Answer: C"
        assert model.invoke("Test 3") == "Answer: D"
        
        # Should cycle back to first response
        assert model.invoke("Test 4") == "Answer: B"
    
    def test_get_config(self):
        """Test getting mock model configuration."""
        responses = ["A", "B"]
        model = MockLanguageModel(responses)
        
        config = model.get_config()
        
        assert config["provider"] == "mock"
        assert config["model"] == "mock-model"
        assert config["responses_count"] == 2


class TestOllamaLanguageModel:
    """Tests for OllamaLanguageModel class."""
    
    def test_config_initialization(self):
        """Test model initialization with config."""
        config = ModelConfig(
            model_name="test-model",
            base_url="http://test:1234",
            temperature=0.5,
            max_tokens=20
        )
        
        model = OllamaLanguageModel(config)
        
        assert model.config.model_name == "test-model"
        assert model.config.base_url == "http://test:1234"
        assert model.config.temperature == 0.5
    
    def test_get_config(self):
        """Test getting model configuration."""
        config = ModelConfig(
            model_name="test-model",
            temperature=0.7
        )
        
        model = OllamaLanguageModel(config)
        model_config = model.get_config()
        
        assert model_config["provider"] == "ollama"
        assert model_config["model"] == "test-model"
        assert model_config["temperature"] == 0.7
    
    @patch('testing_harness.models.OllamaLLM')
    def test_invoke(self, mock_ollama_llm):
        """Test model invocation with mocked LLM."""
        # Setup mock
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = "  Answer: B  "
        mock_ollama_llm.return_value = mock_llm_instance
        
        config = ModelConfig(model_name="test-model")
        model = OllamaLanguageModel(config)
        
        response = model.invoke("Test prompt")
        
        # Should strip whitespace
        assert response == "Answer: B"
        
        # Verify LLM was called correctly
        mock_ollama_llm.assert_called_once_with(
            model="test-model",
            base_url="http://localhost:11434",
            temperature=0.0,
            num_predict=10,
            stop=["\n"]
        )
        mock_llm_instance.invoke.assert_called_once_with("Test prompt")
    
    @patch('testing_harness.models.OllamaLLM')
    def test_llm_reuse(self, mock_ollama_llm):
        """Test that LLM instance is reused across calls."""
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = "Response"
        mock_ollama_llm.return_value = mock_llm_instance
        
        config = ModelConfig()
        model = OllamaLanguageModel(config)
        
        # Make multiple calls
        model.invoke("Prompt 1")
        model.invoke("Prompt 2")
        
        # LLM should only be created once
        mock_ollama_llm.assert_called_once()
        
        # But invoke should be called twice
        assert mock_llm_instance.invoke.call_count == 2


class TestCreateLanguageModel:
    """Tests for create_language_model factory function."""
    
    def test_create_ollama_model(self):
        """Test creating Ollama model."""
        config = ModelConfig(provider="ollama", model_name="test-model")
        
        model = create_language_model(config, async_mode=False)
        
        assert isinstance(model, OllamaLanguageModel)
        assert model.config.model_name == "test-model"
    
    def test_create_unsupported_provider(self):
        """Test creating model with unsupported provider."""
        config = ModelConfig(provider="unsupported")
        
        with pytest.raises(ValueError, match="Unsupported model provider"):
            create_language_model(config)
    
    @patch('testing_harness.models.AsyncOllamaLanguageModel')
    def test_create_async_model(self, mock_async_model):
        """Test creating async model."""
        config = ModelConfig(provider="ollama")
        
        create_language_model(config, async_mode=True)
        
        mock_async_model.assert_called_once_with(config)