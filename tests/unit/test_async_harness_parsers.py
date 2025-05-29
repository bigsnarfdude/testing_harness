import pytest
import pandas as pd
from scripts.async_harness import (
    parse_csv_row,
    prepare_options,
    parse_model_response,
    AnswerChoice,
    QuestionOptions,
    ModelResponse
)

# Tests for parse_csv_row
def test_parse_csv_row_valid():
    data = ["What is 2+2?", "3", "4", "5", "6", "B"]
    series = pd.Series(data, index=['question', 'opt_a', 'opt_b', 'opt_c', 'opt_d', 'correct'])
    question, options, correct_letter = parse_csv_row(series)
    assert question == "What is 2+2?"
    assert options == ["3", "4", "5", "6"]
    assert correct_letter == "B"

def test_parse_csv_row_missing_option():
    # Simulate a row where an option might be missing or NaN, then filled by iloc with 'nan'
    data = ["What is 2+2?", "3", "4", pd.NA, "6", "B"] 
    series = pd.Series(data, index=['question', 'opt_a', 'opt_b', 'opt_c', 'opt_d', 'correct'])
    # parse_csv_row converts NA to "nan" string
    question, options, correct_letter = parse_csv_row(series)
    assert question == "What is 2+2?"
    assert options == ["3", "4", "nan", "6"] 
    assert correct_letter == "B"


def test_parse_csv_row_invalid_correct_letter():
    data = ["What is the capital of France?", "Paris", "London", "Berlin", "Rome", "E"]
    series = pd.Series(data, index=['question', 'opt_a', 'opt_b', 'opt_c', 'opt_d', 'correct'])
    with pytest.raises(ValueError, match="Invalid correct answer letter: E"):
        parse_csv_row(series)

# Tests for prepare_options
def test_prepare_options_valid():
    question_data = (
        "What is the color of the sky?",
        ["Red", "Green", "Blue", "Yellow"],
        "C"
    )
    question_options = prepare_options(question_data)
    assert question_options.correct_letter == AnswerChoice.C
    assert len(question_options.options) == 4
    assert question_options.options[0] == (AnswerChoice.A, "Red")
    assert question_options.options[1] == (AnswerChoice.B, "Green")
    assert question_options.options[2] == (AnswerChoice.C, "Blue")
    assert question_options.options[3] == (AnswerChoice.D, "Yellow")
    assert question_options.letter_option_map == {
        AnswerChoice.A: "Red",
        AnswerChoice.B: "Green",
        AnswerChoice.C: "Blue",
        AnswerChoice.D: "Yellow",
    }

# Tests for parse_model_response
@pytest.mark.parametrize("response_text, expected_choice", [
    ("Answer: A", AnswerChoice.A),
    ("Answer : B", AnswerChoice.B),
    ("  Answer:C  ", AnswerChoice.C),
    ("Answer is D", AnswerChoice.D),
    ("The correct answer is A.", AnswerChoice.A),
    ("A is the answer", AnswerChoice.A), # Fallback
    ("Final Answer: B", AnswerChoice.B),
    ("The final answer is (C)", AnswerChoice.C), # Fallback for "(C)"
    ("My choice is D", AnswerChoice.D), # Fallback
    ("It has to be A", AnswerChoice.A), # Fallback
    ("I think it's B.", AnswerChoice.B), # Fallback
    ("The choice is C, definitely.", AnswerChoice.C), # Fallback
    ("The option is D", AnswerChoice.D), # Fallback
    ("A", AnswerChoice.A), # Fallback, direct letter
    (" (B) ", AnswerChoice.B), # Fallback, letter in parens
    ("The letter C is correct.", AnswerChoice.C) # Fallback
])
def test_parse_model_response_valid_formats(response_text, expected_choice):
    model_response = parse_model_response(response_text)
    assert model_response.answer == expected_choice

@pytest.mark.parametrize("response_text", [
    "I don't know the answer.",
    "This is a difficult question.",
    "Answer: E",
    "Answer: AB",
    "Choice: Z"
])
def test_parse_model_response_invalid_formats(response_text):
    with pytest.raises(ValueError, match="Could not parse model response"):
        parse_model_response(response_text)

def test_parse_model_response_empty():
    with pytest.raises(ValueError, match="Could not parse model response"):
        parse_model_response("")

def test_parse_model_response_only_whitespace():
    with pytest.raises(ValueError, match="Could not parse model response"):
        parse_model_response("    \n   ")

# Test to ensure ModelResponse correctly validates enum members
def test_model_response_validation():
    response = ModelResponse(answer=AnswerChoice.A)
    assert response.answer == AnswerChoice.A

    with pytest.raises(ValueError): # Pydantic uses ValueError for enum validation
        ModelResponse(answer="E")

# Test QuestionOptions correct_letter validation
def test_question_options_correct_letter_validation():
    with pytest.raises(ValueError): # Pydantic uses ValueError for enum validation
        QuestionOptions(
            options=[(AnswerChoice.A, "Option A")],
            correct_letter="Z", # Invalid
            letter_option_map={AnswerChoice.A: "Option A"}
        )
