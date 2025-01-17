# testing-harness.py
import glob
import json
import logging
import os
import re
import time
import argparse
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
import asyncio

import pandas as pd
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field, ConfigDict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

class Config:
    """Configuration settings for the testing harness."""
    MAX_RETRIES = 4
    RATE_LIMIT_DELAY = 1.0
    MAX_WORKERS = 4
    BATCH_SIZE = 5
    REQUEST_TIMEOUT = 30.0
    RETRY_DELAY = 2.0
    CSV_PATTERN = "*.csv"
    CONSOLE = Console()

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_number": self.question_number,
            "question": self.question,
            "options": self.options.dict(),
            "model_answer": self.model_answer.dict() if self.model_answer else None,
            "is_correct": self.is_correct,
            "error": self.error,
            "duration": self.duration,
            "retries": self.retries,
            "timestamp": self.timestamp.isoformat()
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
                "timestamp": result.timestamp.isoformat()
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
        total = len(self.test_cases)
        passed = sum(1 for tc in self.test_cases if tc.passed)
        failed = total - passed
        
        return {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed,
            "accuracy": (passed / total * 100) if total > 0 else 0,
            "average_duration": sum(tc.duration or 0 for tc in self.test_cases) / total if total > 0 else 0,
            "total_retries": sum(tc.retries for tc in self.test_cases),
            "error_rate": (len([tc for tc in self.test_cases if tc.error_messages]) / total * 100) if total > 0 else 0
        }

class AsyncLanguageModel(ABC):
    """Abstract base class for async language models."""
    
    @abstractmethod
    async def ainvoke(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    async def aclose(self):
        pass

class AsyncOllamaLanguageModel(AsyncLanguageModel):
    """Async implementation for Ollama language model."""
    
    def __init__(self, model: str, base_url: str, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs
        self._llm = None
    
    async def ainvoke(self, prompt: str) -> str:
        if not self._llm:
            self._llm = OllamaLLM(
                model=self.model,
                base_url=self.base_url,
                **self.kwargs
            )
        return await asyncio.to_thread(self._llm.invoke, prompt)
    
    async def aclose(self):
        self._llm = None

def parse_csv_row(row: pd.Series) -> Tuple[str, List[str], str]:
    """Parse a row from the CSV file."""
    question = str(row.iloc[0])
    options = [str(row.iloc[i]) for i in range(1, 5)]
    correct_letter = str(row.iloc[-1]).strip().upper()
    if correct_letter not in AnswerChoice.__members__:
        raise ValueError(f"Invalid correct answer letter: {correct_letter}")
    return question, options, correct_letter

def prepare_options(question_data: Tuple[str, List[str], str]) -> QuestionOptions:
    """Prepare question options."""
    _, answer_choices, correct_letter = question_data
    options = list(zip(
        [AnswerChoice.A, AnswerChoice.B, AnswerChoice.C, AnswerChoice.D],
        answer_choices
    ))
    letter_option_map = {letter: option.strip() for letter, option in options}
    
    return QuestionOptions(
        options=options,
        correct_letter=AnswerChoice(correct_letter),
        letter_option_map=letter_option_map
    )

def parse_model_response(response: str) -> ModelResponse:
    """Parse the model's response."""
    match = re.search(r"Answer\s*[:\-is]+\s*([A-D])\b", response, re.IGNORECASE | re.MULTILINE)
    
    if not match:
        fallback_match = re.search(r"\b([A-D])\b", response, re.IGNORECASE)
        if fallback_match:
            answer = fallback_match.group(1).upper()
            logging.debug(f"Fallback parsing successful: Found '{answer}' in response.")
            return ModelResponse(answer=AnswerChoice(answer))
        
        logging.debug(f"Unparsable model response: '{response}'")
        raise ValueError("Could not parse model response")
    
    return ModelResponse(answer=AnswerChoice(match.group(1).upper()))

class TestHarness:
    """Main test harness class."""
    
    def __init__(self, model: AsyncLanguageModel, output_dir: str, verbose: bool = False):
        self.model = model
        self.output_dir = output_dir
        self.verbose = verbose
        self.console = Config.CONSOLE
        os.makedirs(output_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        from logging.handlers import RotatingFileHandler
        
        log_file = os.path.join(self.output_dir, f"test_run_{datetime.now():%Y%m%d_%H%M%S}.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    async def process_question_batch(
        self,
        questions: List[Tuple[pd.Series, int]]
    ) -> List[EvaluationResult]:
        """Process a batch of questions concurrently."""
        tasks = [
            self.evaluate_question(row, qnum)
            for row, qnum in questions
        ]
        return await asyncio.gather(*tasks)

    async def evaluate_question(
        self,
        row: pd.Series,
        question_number: int,
        retry_count: int = 0
    ) -> EvaluationResult:
        """Evaluate a single question."""
        start_time = time.time()
        result = EvaluationResult(
            question_number=question_number,
            question=str(row.iloc[0]),
            options=QuestionOptions(
                options=[],
                correct_letter=AnswerChoice.A,
                letter_option_map={}
            )
        )
        
        try:
            question_data = parse_csv_row(row)
            result.options = prepare_options(question_data)
            
            response = await self.model.ainvoke(self._create_prompt(
                question_data[0],
                result.options
            ))
            
            model_response = parse_model_response(response)
            result.model_answer = model_response
            result.is_correct = model_response.answer == result.options.correct_letter
            
        except Exception as e:
            if retry_count < Config.MAX_RETRIES:
                await asyncio.sleep(Config.RETRY_DELAY)
                return await self.evaluate_question(
                    row, question_number, retry_count + 1
                )
            result.error = str(e)
            logging.error(f"Failed to evaluate question {question_number}: {e}")
            
        finally:
            result.duration = time.time() - start_time
            result.retries = retry_count
        
        return result

    def _create_prompt(self, question: str, options: QuestionOptions) -> str:
        """Create the prompt for the model."""
        options_text = "\n".join(
            f"{letter}. {option}" for letter, option in options.options
        )
        return (
            f"Question: {question}\n"
            f"Options:\n{options_text}\n\n"
            "Please provide the answer in the following exact format and nothing else:\n"
            "Answer: <LETTER>\n"
            "Where <LETTER> is one of A, B, C, or D."
        )


    async def run(self, csv_dir: str) -> TestSuite:
        """Run the test harness."""
        csv_files = glob.glob(os.path.join(csv_dir, Config.CSV_PATTERN))
        if not csv_files:
            raise ValueError(f"No CSV files found in {csv_dir}")
        
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            for csv_file in csv_files:
                task_id = progress.add_task(
                    f"Processing {os.path.basename(csv_file)}",
                    total=100
                )
                
                try:
                    # Read CSV with explicit column names and handle potential file reading errors
                    df = pd.read_csv(
                        csv_file,
                        header=None,
                        names=['question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_answer'],
                        on_bad_lines='warn'
                    )
                    
                    # Validate DataFrame structure
                    required_columns = 6  # question + 4 options + correct answer
                    if df.shape[1] != required_columns:
                        raise ValueError(
                            f"CSV file {csv_file} has {df.shape[1]} columns, "
                            f"expected {required_columns} columns"
                        )
                    
                    # Clean the data
                    df = df.dropna()  # Remove rows with missing values
                    df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
                    
                    # Validate correct answer column
                    invalid_answers = df.iloc[:, -1].apply(
                        lambda x: str(x).upper().strip() not in AnswerChoice.__members__
                    )
                    if invalid_answers.any():
                        invalid_rows = df[invalid_answers].index.tolist()
                        logging.warning(
                            f"Skipping rows {invalid_rows} in {csv_file} due to "
                            "invalid correct answer values"
                        )
                        df = df[~invalid_answers]
                    
                    questions = list(enumerate(df.iterrows(), 1))
                    
                    for i in range(0, len(questions), Config.BATCH_SIZE):
                        batch = questions[i:i + Config.BATCH_SIZE]
                        results = await self.process_question_batch(
                            [(row, qnum) for qnum, (_, row) in batch]
                        )
                        all_results.extend(results)
                        
                        progress.update(
                            task_id,
                            completed=(i + len(batch)) / len(questions) * 100
                        )
                        
                        await asyncio.sleep(Config.RATE_LIMIT_DELAY)
                
                except Exception as e:
                    logging.error(f"Error processing file {csv_file}: {str(e)}")
                    continue
        
        if not all_results:
            raise ValueError("No valid results were generated from any CSV files")
        
        return self._create_test_suite(all_results)


    def _create_test_suite(self, results: List[EvaluationResult]) -> TestSuite:
        """Create a test suite from results."""
        test_cases = [TestCase.from_evaluation_result(result) for result in results]
        
        return TestSuite(
            name="Multiple Choice Evaluation",
            description="Evaluation of model performance on multiple choice questions",
            test_cases=test_cases,
            metadata={
                "timestamp": datetime.now().isoformat(),
                **self._calculate_analytics(test_cases)
            }
        )

    def _calculate_analytics(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Calculate analytics for the test suite."""
        total = len(test_cases)
        passed = sum(1 for tc in test_cases if tc.passed)
        
        return {
            "total_questions": total,
            "passed_tests": passed,
            "accuracy": (passed / total * 100) if total > 0 else 0,
            "average_duration": sum(tc.duration or 0 for tc in test_cases) / total if total > 0 else 0,
            "error_rate": (len([tc for tc in test_cases if tc.error_messages]) / total * 100) if total > 0 else 0
        }

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced testing harness for multiple choice questions."
    )
    parser.add_argument(
        "csv_dir",
        type=str,
        help="Directory containing CSV files with questions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:latest",
        help="Model to use for evaluation"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for the Ollama API"
    )
    
    args = parser.parse_args()
    
    model = AsyncOllamaLanguageModel(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
        max_tokens=10,
        stop_sequences=["\n"]
    )
    
    harness = TestHarness(
        model=model,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    try:
        test_suite = await harness.run(args.csv_dir)
        analytics = test_suite.get_analytics()
        
        # Display results using rich
        table = Table(title="Test Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in analytics.items():
            table.add_row(
                key.replace("_", " ").title(),
                f"{value:.2f}" if isinstance(value, float) else str(value)
            )
        
        Config.CONSOLE.print(table)
        
    finally:
        await model.aclose()

if __name__ == "__main__":
    asyncio.run(main())