"""Base classes and models for the testing harness."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict


class AnswerChoice(str, Enum):
    """Enumeration of possible answer choices."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class QuestionOptions(BaseModel):
    """Model for question options and correct answer."""
    model_config = ConfigDict(protected_namespaces=())
    
    options: List[Tuple[AnswerChoice, str]]
    correct_letter: AnswerChoice
    letter_option_map: Dict[AnswerChoice, str]


class ModelResponse(BaseModel):
    """Model for the language model's response."""
    answer: AnswerChoice = Field(..., description="The model's answer (A, B, C, or D)")
    raw_response: str = Field(..., description="The raw response from the model")
    confidence: Optional[float] = Field(None, description="Confidence score if available")
    
    @classmethod
    def validate_answer(cls, v):
        if not isinstance(v, AnswerChoice):
            raise ValueError(f"Invalid answer choice: {v}")
        return v


class EvaluationResult(BaseModel):
    """Model for evaluation results."""
    model_config = ConfigDict(protected_namespaces=())
    
    question_number: int
    question: str
    options: QuestionOptions
    model_answer: Optional[ModelResponse] = None
    is_correct: bool = False
    error: Optional[str] = None
    duration: Optional[float] = None
    retries: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question_number": self.question_number,
            "question": self.question,
            "options": self.options.model_dump(),
            "model_answer": self.model_answer.model_dump() if self.model_answer else None,
            "is_correct": self.is_correct,
            "error": self.error,
            "duration": self.duration,
            "retries": self.retries,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class TestCase(BaseModel):
    """Model for test cases."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str
    type: str
    input: str
    expected_output: str
    actual_output: Optional[str] = None
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration: Optional[float] = None
    retries: int = 0
    error_messages: List[str] = Field(default_factory=list)
    
    @classmethod
    def from_evaluation_result(cls, result: EvaluationResult) -> 'TestCase':
        """Create a TestCase from an EvaluationResult."""
        return cls(
            id=f"q{result.question_number}",
            type="multiple_choice",
            input=result.question,
            expected_output=str(result.options.correct_letter),
            actual_output=str(result.model_answer.answer) if result.model_answer else None,
            passed=result.is_correct,
            metadata={
                "options": result.options.letter_option_map,
                "error": result.error,
                "timestamp": result.timestamp.isoformat(),
                **result.metadata,
            },
            duration=result.duration,
            retries=result.retries,
            error_messages=[result.error] if result.error else []
        )


class TestSuite(BaseModel):
    """Model for test suites."""
    model_config = ConfigDict(protected_namespaces=())
    
    name: str
    description: str
    test_cases: List[TestCase]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Calculate analytics for the test suite."""
        total = len(self.test_cases)
        passed = sum(1 for tc in self.test_cases if tc.passed)
        failed = total - passed
        
        total_duration = sum(tc.duration or 0 for tc in self.test_cases)
        errors_count = len([tc for tc in self.test_cases if tc.error_messages])
        
        return {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed,
            "accuracy": (passed / total * 100) if total > 0 else 0,
            "average_duration": total_duration / total if total > 0 else 0,
            "total_duration": total_duration,
            "total_retries": sum(tc.retries for tc in self.test_cases),
            "error_rate": (errors_count / total * 100) if total > 0 else 0,
        }
    
    def save_results(self, output_dir: str, format: str = "json") -> str:
        """Save test results to file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            output_file = os.path.join(output_dir, f"test_results_{timestamp}.json")
            with open(output_file, 'w') as f:
                json.dump(self.model_dump(), f, indent=2)
        elif format == "csv":
            import pandas as pd
            output_file = os.path.join(output_dir, f"test_results_{timestamp}.csv")
            
            # Convert test cases to DataFrame
            data = []
            for tc in self.test_cases:
                data.append({
                    "id": tc.id,
                    "question": tc.input,
                    "expected": tc.expected_output,
                    "actual": tc.actual_output,
                    "passed": tc.passed,
                    "duration": tc.duration,
                    "retries": tc.retries,
                    "error": "; ".join(tc.error_messages) if tc.error_messages else None,
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_file


class LanguageModel(ABC):
    """Abstract base class for language models."""
    
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Invoke the model with a prompt."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        pass


class AsyncLanguageModel(ABC):
    """Abstract base class for async language models."""
    
    @abstractmethod
    async def ainvoke(self, prompt: str) -> str:
        """Async invoke the model with a prompt."""
        pass
    
    @abstractmethod
    async def aclose(self):
        """Close any resources."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        pass


class BaseTestHarness(ABC):
    """Base class for test harnesses."""
    
    def __init__(self, model: LanguageModel, output_dir: str, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.output_dir = output_dir
        self.config = config or {}
        os.makedirs(output_dir, exist_ok=True)
        self._checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    
    @abstractmethod
    def evaluate_question(self, question_data: Any, question_number: int) -> EvaluationResult:
        """Evaluate a single question."""
        pass
    
    @abstractmethod
    def run(self, data_source: str) -> TestSuite:
        """Run the test harness on a data source."""
        pass
    
    def save_checkpoint(self, results: List[EvaluationResult], current_index: int):
        """Save checkpoint for resuming interrupted runs."""
        checkpoint_data = {
            "current_index": current_index,
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
            "model_config": self.model.get_config(),
        }
        
        with open(self._checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Tuple[int, List[EvaluationResult]]]:
        """Load checkpoint if it exists."""
        if not os.path.exists(self._checkpoint_file):
            return None
        
        try:
            with open(self._checkpoint_file, 'r') as f:
                data = json.load(f)
            
            results = []
            for r_dict in data["results"]:
                # Reconstruct EvaluationResult from dict
                # This is simplified - you'd need proper deserialization
                results.append(EvaluationResult(**r_dict))
            
            return data["current_index"], results
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self):
        """Clear checkpoint file after successful completion."""
        if os.path.exists(self._checkpoint_file):
            os.remove(self._checkpoint_file)