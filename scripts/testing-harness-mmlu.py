# python testing-harness-mmlu.py mmlu --output-dir logs
import glob
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any

import pandas as pd
from langchain_ollama import OllamaLLM  # Ensure this package is installed
from pydantic import BaseModel, Field, ConfigDict

# Constants
MAX_RETRIES = 4
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
RATE_LIMIT_DELAY = 1.0  # Delay in seconds between API calls


class TestCase(BaseModel):
    """Test case model for storing test data."""
    model_config = ConfigDict(protected_namespaces=())

    id: str
    type: str
    input: str
    expected_output: str
    actual_output: Optional[str] = None
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestSuite(BaseModel):
    """Test suite model for organizing test cases."""
    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: str
    test_cases: List[TestCase]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class QuestionOptions(BaseModel):
    options: List[Tuple[AnswerChoice, str]]
    correct_letter: AnswerChoice
    letter_option_map: Dict[AnswerChoice, str]


class ModelResponse(BaseModel):
    answer: AnswerChoice = Field(..., description="The model's answer (A, B, C, or D)")

    @classmethod
    def validate_answer(cls, v):
        if not isinstance(v, AnswerChoice):
            raise ValueError(f"Invalid answer choice: {v}")
        return v


class EvaluationResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    question_number: int
    question: str
    options: QuestionOptions
    model_answer: Optional[ModelResponse] = None
    is_correct: bool = False
    error: Optional[str] = None


def setup_logging(log_dir: str = "logs", verbose: bool = False):
    """Set up logging configuration with optional verbose output."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_run_{timestamp}.log")

    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_test_suite(file_path: str) -> TestSuite:
    """Load a test suite from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return TestSuite(**data)


def save_test_results(test_suite: TestSuite, output_dir: str):
    """Save test results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"test_results_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(test_suite.model_dump(), f, indent=2)


def parse_csv_row(row: pd.Series) -> Tuple[str, List[str], str]:
    """Parse a row from the CSV and extract the question, options, and correct answer."""
    question = str(row.iloc[0])
    options = [str(row.iloc[i]) for i in range(1, 5)]
    correct_letter = str(row.iloc[-1]).strip().upper()
    if correct_letter not in AnswerChoice.__members__:
        raise ValueError(f"Invalid correct answer letter: {correct_letter}")
    return question, options, correct_letter


def prepare_options(question_data: Tuple[str, List[str], str]) -> QuestionOptions:
    """Prepare the QuestionOptions from parsed CSV data."""
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
    """
    Parse the model's response to extract the answer choice.
    Enhanced to handle more variations and log unparsed responses.
    """
    # Improved regex to capture different separators and possible trailing characters
    match = re.search(r"Answer\s*[:\-is]+\s*([A-D])\b", response, re.IGNORECASE | re.MULTILINE)

    if not match:
        # Attempt a fallback: find any standalone A-D letter
        fallback_match = re.search(r"\b([A-D])\b", response, re.IGNORECASE)
        if fallback_match:
            answer = fallback_match.group(1).upper()
            logging.debug(f"Fallback parsing successful: Found '{answer}' in response.")
            return ModelResponse(answer=AnswerChoice(answer))

        # If still no match, log the entire response for debugging
        logging.debug(f"Unparsable model response: '{response}'")
        raise ValueError("Could not parse model response")

    return ModelResponse(answer=AnswerChoice(match.group(1).upper()))


class LanguageModel(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass


class OllamaLanguageModel(LanguageModel):
    """Concrete class for Ollama."""

    def __init__(self, model: str, base_url: str, temperature: float = 0.0, max_tokens: int = 10, stop_sequences: Optional[List[str]] = None):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences or ["\n"]

    def invoke(self, prompt: str) -> str:
        llm = OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_sequences=self.stop_sequences
        )
        response = llm.invoke(prompt).strip()
        return response


def evaluate_question(
    row: pd.Series, 
    question_number: int, 
    language_model: LanguageModel, 
    retry_count: int = 0
) -> EvaluationResult:
    """Evaluate a single question using the language model."""
    if retry_count >= MAX_RETRIES:
        logging.error(f"Max retries reached for question {question_number}.")
        return EvaluationResult(
            question_number=question_number,
            question="Error parsing question",
            options=QuestionOptions(
                options=[],
                correct_letter=AnswerChoice.A,
                letter_option_map={}
            ),
            error="Max retries reached",
        )

    try:
        question_data = parse_csv_row(row)
    except ValueError as ve:
        logging.error(f"Error parsing CSV row for question {question_number}: {ve}")
        return EvaluationResult(
            question_number=question_number,
            question=str(row.iloc[0]),
            options=QuestionOptions(
                options=[],
                correct_letter=AnswerChoice.A,
                letter_option_map={}
            ),
            error=str(ve)
        )

    question = question_data[0]
    options = prepare_options(question_data)

    result = EvaluationResult(
        question_number=question_number,
        question=question,
        options=options
    )

    try:
        options_text = "\n".join([f"{letter}. {option}" for letter, option in options.options])
        prompt = (
            f"Question: {question}\n"
            f"Options:\n{options_text}\n\n"
            "Please provide the answer in the following exact format and nothing else:\n"
            "Answer: <LETTER>\n"
            "Where <LETTER> is one of A, B, C, or D."
        )
        response = language_model.invoke(prompt)

        logging.debug(f"Model response for question {question_number}: '{response}'")

        model_response = parse_model_response(response)
        result.model_answer = model_response
        result.is_correct = model_response.answer == options.correct_letter

    except Exception as e:
        logging.warning(f"Error evaluating question {question_number}: {e}. Retrying...")
        return evaluate_question(row, question_number, language_model, retry_count + 1)

    log_evaluation_result(result)
    return result


def log_evaluation_result(result: EvaluationResult):
    """Log the details of a single evaluation result."""
    if result.is_correct:
        logging.info(f"Question {result.question_number}: Correct")
    else:
        if result.model_answer:
            logging.info(
                f"Question {result.question_number}: Incorrect - "
                f"Model answered {result.model_answer.answer} "
                f"({result.options.letter_option_map.get(result.model_answer.answer, 'N/A')}), "
                f"correct answer was {result.options.correct_letter} "
                f"({result.options.letter_option_map.get(result.options.correct_letter, 'N/A')})"
            )
        else:
            logging.error(f"Question {result.question_number}: Error - {result.error}")


def print_evaluation_result(result: EvaluationResult):
    """Print the evaluation result to the console."""
    print(f"\nQuestion {result.question_number}:")
    print(f"Q: {result.question}")
    print("Options:")
    for letter, text in result.options.options:
        print(f"{letter}. {text}")

    if result.model_answer:
        answer = result.model_answer.answer
        print(f"Model answered: {answer} ({result.options.letter_option_map.get(answer, 'N/A')})")
        print(f"Correct answer: {result.options.correct_letter} "
              f"({result.options.letter_option_map.get(result.options.correct_letter, 'N/A')})")
        print(f"Result: {'✓ Correct' if result.is_correct else '✗ Incorrect'}")
    else:
        print(f"Error: {result.error}")


def convert_evaluation_to_test_case(result: EvaluationResult) -> TestCase:
    """Convert an EvaluationResult to a TestCase."""
    return TestCase(
        id=f"q{result.question_number}",
        type="multiple_choice",
        input=result.question,
        expected_output=str(result.options.correct_letter),
        actual_output=str(result.model_answer.answer) if result.model_answer else None,
        passed=result.is_correct,
        metadata={
            "options": result.options.letter_option_map,
            "error": result.error
        }
    )


def process_csv_files(csv_dir: str, language_model: LanguageModel) -> TestSuite:
    """Process all CSV files in the directory and return a TestSuite."""
    csv_files = glob.glob(f"{csv_dir}/*.csv")
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {csv_dir}")

    all_results = []

    for csv_file in csv_files:
        logging.info(f"\nProcessing file: {csv_file}")
        df = pd.read_csv(csv_file, header=None)

        for i, row in df.iterrows():
            question_number = i + 1
            result = evaluate_question(row, question_number, language_model)
            all_results.append(result)
            print_evaluation_result(result)
            time.sleep(RATE_LIMIT_DELAY)  # Rate limiting to prevent overwhelming the API

    test_cases = [convert_evaluation_to_test_case(result) for result in all_results]

    passed_tests = sum(1 for tc in test_cases if tc.passed)
    total_tests = len(test_cases)

    return TestSuite(
        name="Multiple Choice Evaluation",
        description="Evaluation of model performance on multiple choice questions",
        test_cases=test_cases,
        metadata={
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(test_cases),
            "passed_tests": passed_tests,
            "accuracy": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }
    )


def generate_summary_report(test_suite: TestSuite, output_dir: str):
    """Generate a summary report as a CSV file."""
    summary = {
        "Timestamp": test_suite.metadata["timestamp"],
        "Total Questions": test_suite.metadata["total_questions"],
        "Passed Tests": test_suite.metadata["passed_tests"],
        "Accuracy (%)": f"{test_suite.metadata['accuracy']:.2f}"
    }

    summary_file = os.path.join(output_dir, "summary_report.csv")
    with open(summary_file, 'w') as f:
        f.write("Metric,Value\n")
        for key, value in summary.items():
            f.write(f"{key},{value}\n")

    logging.info(f"Summary report generated at: {summary_file}")


def main(csv_dir: str, output_dir: str = "results", verbose: bool = False):
    """Main function to run the testing harness."""
    setup_logging(verbose=verbose)

    try:
        # Initialize the language model with adjusted parameters
        language_model = OllamaLanguageModel(
            model="llama3.2:latest",
            base_url="http://localhost:11434",
            temperature=0.0,          # Ensures deterministic output
            max_tokens=10,            # Limits response length
            stop_sequences=["\n"]     # Stops after the first newline
        )

        test_suite = process_csv_files(csv_dir, language_model)

        save_test_results(test_suite, output_dir)
        generate_summary_report(test_suite, output_dir)

        # Log and print the summary
        total = test_suite.metadata["total_questions"]
        passed = test_suite.metadata["passed_tests"]
        accuracy = test_suite.metadata["accuracy"]

        summary_message = (
            f"\n=== Test Suite Summary ===\n"
            f"Total Questions: {total}\n"
            f"Passed Tests: {passed}\n"
            f"Accuracy: {accuracy:.2f}%\n"
            f"Summary report saved to: {os.path.join(output_dir, 'summary_report.csv')}\n"
        )

        logging.info(summary_message)
        print(summary_message)

    except Exception as e:
        logging.error(f"Error running test harness: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the testing harness for multiple choice questions.")
    parser.add_argument(
        "csv_dir",
        type=str,
        help="Directory containing CSV files with questions to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save test results (default: results)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging"
    )

    args = parser.parse_args()
    main(args.csv_dir, args.output_dir, args.verbose)