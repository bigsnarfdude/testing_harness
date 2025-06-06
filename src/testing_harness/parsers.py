"""Parsers for handling different data formats and model responses."""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd

from .base import AnswerChoice, QuestionOptions, ModelResponse, EvaluationResult


class ResponseParser:
    """Parser for model responses with multiple fallback strategies."""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.patterns = [
            # Primary pattern: "Answer: X" or "Answer is X"
            r"Answer\s*[:\-is]+\s*([A-D])\b",
            # Secondary: "The answer is X"
            r"The\s+answer\s+is\s*[:\-]?\s*([A-D])\b",
            # Tertiary: "X)" or "(X)" at the start of a line
            r"^\s*\(?([A-D])\)?\s*",
            # Fallback: Any standalone letter
            r"\b([A-D])\b",
        ]
    
    def parse(self, response: str) -> ModelResponse:
        """Parse model response with multiple strategies."""
        response = response.strip()
        
        # Try each pattern in order
        for i, pattern in enumerate(self.patterns):
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).upper()
                logging.debug(f"Pattern {i+1} matched: Found '{answer}' in response")
                
                # Calculate confidence based on which pattern matched
                confidence = 1.0 - (i * 0.2)  # Primary pattern = 1.0, fallback = 0.4
                
                return ModelResponse(
                    answer=AnswerChoice(answer),
                    raw_response=response,
                    confidence=confidence
                )
        
        # If strict mode, raise error; otherwise try to extract any letter
        if self.strict:
            logging.error(f"Could not parse response: {response}")
            raise ValueError("Could not parse model response")
        
        # Last resort: find any A-D letter
        letters = re.findall(r"[A-D]", response, re.IGNORECASE)
        if letters:
            answer = letters[0].upper()
            logging.warning(f"Last resort parsing: Using first letter '{answer}' found")
            return ModelResponse(
                answer=AnswerChoice(answer),
                raw_response=response,
                confidence=0.2
            )
        
        raise ValueError("No valid answer choice found in response")


class CSVParser:
    """Parser for CSV files containing questions."""
    
    def __init__(self, format: str = "standard"):
        """
        Initialize CSV parser.
        
        Args:
            format: CSV format type ('standard', 'gpqa', 'mmlu')
        """
        self.format = format
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse CSV file and return list of question data."""
        if self.format == "gpqa":
            return self._parse_gpqa_format(file_path)
        elif self.format == "mmlu":
            return self._parse_mmlu_format(file_path)
        else:
            return self._parse_standard_format(file_path)
    
    def _parse_standard_format(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse standard CSV format (question, 4 options, correct answer)."""
        df = pd.read_csv(
            file_path,
            header=None,
            names=['question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_answer'],
            on_bad_lines='warn'
        )
        
        # Clean and validate data
        df = df.dropna()
        df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
        
        questions = []
        for idx, row in df.iterrows():
            try:
                question_data = self._parse_standard_row(row)
                questions.append(question_data)
            except ValueError as e:
                logging.warning(f"Skipping row {idx}: {e}")
                continue
        
        return questions
    
    def _parse_standard_row(self, row: pd.Series) -> Dict[str, Any]:
        """Parse a standard format row."""
        question = str(row['question'])
        options = [
            str(row['option_a']),
            str(row['option_b']),
            str(row['option_c']),
            str(row['option_d'])
        ]
        correct_letter = str(row['correct_answer']).strip().upper()
        
        if correct_letter not in AnswerChoice.__members__:
            raise ValueError(f"Invalid correct answer: {correct_letter}")
        
        return {
            "question": question,
            "options": options,
            "correct_letter": correct_letter,
            "format": "standard"
        }
    
    def _parse_gpqa_format(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse GPQA format CSV."""
        df = pd.read_csv(file_path)
        
        questions = []
        for idx, row in df.iterrows():
            try:
                # GPQA has columns: Question, Correct Answer, Incorrect Answer 1-3
                question = str(row["Question"])
                correct = str(row["Correct Answer"])
                incorrect = [
                    str(row["Incorrect Answer 1"]),
                    str(row["Incorrect Answer 2"]),
                    str(row["Incorrect Answer 3"])
                ]
                
                # Randomize options
                import random
                options = [correct] + incorrect
                random.shuffle(options)
                
                # Find correct letter after shuffle
                correct_idx = options.index(correct)
                correct_letter = list(AnswerChoice)[correct_idx].value
                
                questions.append({
                    "question": question,
                    "options": options,
                    "correct_letter": correct_letter,
                    "format": "gpqa"
                })
            except Exception as e:
                logging.warning(f"Error parsing GPQA row {idx}: {e}")
                continue
        
        return questions
    
    def _parse_mmlu_format(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse MMLU format CSV (same as standard but may have subject metadata)."""
        # Extract subject from filename if available
        import os
        subject = os.path.basename(file_path).replace("_val.csv", "").replace("_test.csv", "")
        
        questions = self._parse_standard_format(file_path)
        
        # Add subject metadata
        for q in questions:
            q["subject"] = subject
            q["format"] = "mmlu"
        
        return questions
    
    @staticmethod
    def create_question_options(question_data: Dict[str, Any]) -> QuestionOptions:
        """Create QuestionOptions from parsed question data."""
        options = list(zip(
            [AnswerChoice.A, AnswerChoice.B, AnswerChoice.C, AnswerChoice.D],
            question_data["options"]
        ))
        
        letter_option_map = {letter: option.strip() for letter, option in options}
        
        return QuestionOptions(
            options=options,
            correct_letter=AnswerChoice(question_data["correct_letter"]),
            letter_option_map=letter_option_map
        )