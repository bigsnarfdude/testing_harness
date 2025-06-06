"""Tests for parser modules."""

import pytest
import tempfile
import os
from pathlib import Path

from testing_harness.parsers import ResponseParser, CSVParser
from testing_harness.base import AnswerChoice, ModelResponse


class TestResponseParser:
    """Tests for ResponseParser class."""
    
    def test_parse_standard_format(self):
        """Test parsing standard 'Answer: X' format."""
        parser = ResponseParser()
        
        # Test various formats
        test_cases = [
            ("Answer: A", AnswerChoice.A),
            ("Answer is B", AnswerChoice.B),
            ("The answer: C", AnswerChoice.C),
            ("Answer - D", AnswerChoice.D),
            ("Answer:A", AnswerChoice.A),  # No space
        ]
        
        for response, expected in test_cases:
            result = parser.parse(response)
            assert result.answer == expected
            assert result.raw_response == response
            assert result.confidence >= 0.8  # High confidence for primary patterns
    
    def test_parse_fallback_patterns(self):
        """Test fallback parsing patterns."""
        parser = ResponseParser()
        
        # Test fallback patterns
        test_cases = [
            ("A) This is the correct option", AnswerChoice.A),
            ("(B) Another format", AnswerChoice.B),
            ("Looking at the options, C seems correct", AnswerChoice.C),
            ("I think D", AnswerChoice.D),
        ]
        
        for response, expected in test_cases:
            result = parser.parse(response)
            assert result.answer == expected
            assert result.confidence < 1.0  # Lower confidence for fallback patterns
    
    def test_parse_case_insensitive(self):
        """Test case insensitive parsing."""
        parser = ResponseParser()
        
        test_cases = [
            ("answer: a", AnswerChoice.A),
            ("ANSWER: B", AnswerChoice.B),
            ("Answer: c", AnswerChoice.C),
        ]
        
        for response, expected in test_cases:
            result = parser.parse(response)
            assert result.answer == expected
    
    def test_parse_strict_mode(self):
        """Test strict mode parsing."""
        parser = ResponseParser(strict=True)
        
        # Should work for clear answers
        result = parser.parse("Answer: A")
        assert result.answer == AnswerChoice.A
        
        # Should fail for ambiguous responses
        with pytest.raises(ValueError):
            parser.parse("I'm not sure about this question")
    
    def test_parse_invalid_response(self):
        """Test handling of unparseable responses."""
        parser = ResponseParser(strict=False)
        
        # Should raise error even in non-strict mode if no letters found
        with pytest.raises(ValueError):
            parser.parse("This response has no valid answer choice")
    
    def test_confidence_scores(self):
        """Test confidence score calculation."""
        parser = ResponseParser()
        
        # Primary pattern should have highest confidence
        result1 = parser.parse("Answer: A")
        
        # Fallback pattern should have lower confidence
        result2 = parser.parse("I think it's A")
        
        assert result1.confidence > result2.confidence


class TestCSVParser:
    """Tests for CSVParser class."""
    
    def create_temp_csv(self, content: str) -> str:
        """Create a temporary CSV file with given content."""
        fd, path = tempfile.mkstemp(suffix='.csv')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path
    
    def test_parse_standard_format(self):
        """Test parsing standard CSV format."""
        parser = CSVParser(format="standard")
        
        csv_content = """What is 2+2?,2,3,4,5,C
What color is the sky?,Red,Blue,Green,Yellow,B"""
        
        csv_path = self.create_temp_csv(csv_content)
        
        try:
            questions = parser.parse_file(csv_path)
            
            assert len(questions) == 2
            
            # Check first question
            assert questions[0]["question"] == "What is 2+2?"
            assert questions[0]["options"] == ["2", "3", "4", "5"]
            assert questions[0]["correct_letter"] == "C"
            assert questions[0]["format"] == "standard"
            
            # Check second question
            assert questions[1]["question"] == "What color is the sky?"
            assert questions[1]["correct_letter"] == "B"
            
        finally:
            os.unlink(csv_path)
    
    def test_parse_mmlu_format(self):
        """Test parsing MMLU format (same as standard but with metadata)."""
        parser = CSVParser(format="mmlu")
        
        csv_content = """What is the capital of France?,London,Paris,Berlin,Madrid,B
What is 1+1?,1,2,3,4,B"""
        
        csv_path = self.create_temp_csv("geography_val.csv")
        
        # Write content to the file
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        try:
            questions = parser.parse_file(csv_path)
            
            assert len(questions) == 2
            assert questions[0]["format"] == "mmlu"
            assert questions[0]["subject"] == "geography"
            
        finally:
            os.unlink(csv_path)
    
    def test_create_question_options(self):
        """Test creating QuestionOptions from parsed data."""
        question_data = {
            "question": "Test question?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_letter": "B",
            "format": "standard"
        }
        
        options = CSVParser.create_question_options(question_data)
        
        assert len(options.options) == 4
        assert options.correct_letter == AnswerChoice.B
        assert options.letter_option_map[AnswerChoice.A] == "Option A"
        assert options.letter_option_map[AnswerChoice.B] == "Option B"
    
    def test_handle_invalid_data(self):
        """Test handling of invalid CSV data."""
        parser = CSVParser(format="standard")
        
        # CSV with invalid correct answer
        csv_content = """Question?,A,B,C,D,X"""  # X is not valid
        
        csv_path = self.create_temp_csv(csv_content)
        
        try:
            questions = parser.parse_file(csv_path)
            # Should skip invalid rows
            assert len(questions) == 0
            
        finally:
            os.unlink(csv_path)
    
    def test_handle_missing_data(self):
        """Test handling of CSV with missing data."""
        parser = CSVParser(format="standard")
        
        # CSV with empty cells
        csv_content = """Complete question?,A,B,C,D,A
,A,B,C,D,B
Question with missing option?,A,,C,D,C"""
        
        csv_path = self.create_temp_csv(csv_content)
        
        try:
            questions = parser.parse_file(csv_path)
            # Should only parse the complete question
            assert len(questions) == 1
            assert questions[0]["question"] == "Complete question?"
            
        finally:
            os.unlink(csv_path)