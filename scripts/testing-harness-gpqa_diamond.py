import pandas as pd
import random
import re
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
import sys
import io
import requests
from langchain_ollama import OllamaLLM

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
    
    @field_validator('answer')
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

def download_csv():
    #response = requests.get(url)
    #if response.status_code != 200:
    #    raise requests.HTTPError(f"Failed to download CSV. Status code: {response.status_code}")
    return pd.read_csv('gpqa_diamond.csv')

def prepare_options(row: pd.Series) -> QuestionOptions:
    options = [
        (AnswerChoice.A, row["Correct Answer"]),
        (AnswerChoice.B, row["Incorrect Answer 1"]),
        (AnswerChoice.C, row["Incorrect Answer 2"]),
        (AnswerChoice.D, row["Incorrect Answer 3"]),
    ]
    random.shuffle(options)
    
    letter_option_map = {letter: option for letter, option in options}
    correct_letter = next(letter for letter, option in options if option == row["Correct Answer"])
    
    return QuestionOptions(
        options=options,
        correct_letter=correct_letter,
        letter_option_map=letter_option_map
    )

def parse_model_response(response: str) -> ModelResponse:
    match = re.search(r"Answer:\s*([A-D])\s*$", response, re.IGNORECASE | re.MULTILINE)
    if not match:
        raise ValueError("Could not parse model response")
    return ModelResponse(answer=AnswerChoice(match.group(1).upper()))

def evaluate_question(row: pd.Series, question_number: int, retry_count: int = 0) -> EvaluationResult:
    MAX_RETRIES = 4
    question = row["Question"]
    options = prepare_options(row)
    
    result = EvaluationResult(
        question_number=question_number,
        question=question,
        options=options
    )
    
    try:
        options_text = "\n".join([f"{letter}. {option}" for letter, option in options.options])
        prompt = f"Question: {question}\nOptions:\n{options_text}\n\nPlease output 'Answer: $LETTER' in the last line."
        
        llm = OllamaLLM(model="phi-4_f16:latest", base_url="http://localhost:11434")
        response = llm.invoke(prompt).strip()
        
        model_response = parse_model_response(response)
        result.model_answer = model_response
        result.is_correct = model_response.answer == options.correct_letter
        
    except Exception as e:
        if retry_count < MAX_RETRIES:
            return evaluate_question(row, question_number, retry_count + 1)
        result.error = str(e)
    
    print_evaluation_result(result)
    return result

def print_evaluation_result(result: EvaluationResult):
    print(f"\nQuestion {result.question_number}:")
    print(f"Q: {result.question}")
    print("Options:")
    for letter, text in result.options.options:
        print(f"{letter}. {text}")
    
    if result.model_answer:
        answer = result.model_answer.answer
        print(f"Model answered: {answer} ({result.options.letter_option_map[answer]})")
        print(f"Correct answer: {result.options.correct_letter} "
              f"({result.options.letter_option_map[result.options.correct_letter]})")
        print(f"Result: {'✓ Correct' if result.is_correct else '✗ Incorrect'}")
    else:
        print(f"Error: {result.error}")

def main():
    CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
    df = download_csv()
    print(f"Total questions fetched: {len(df)}")
    
    results = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        result = evaluate_question(row, i)
        results.append(result)
        
        correct_answers = sum(r.is_correct for r in results)
        current_accuracy = (correct_answers / i) * 100
        print(f"Current Accuracy: {current_accuracy:.2f}% ({correct_answers}/{i})")
        print("-" * 80)
    
    final_accuracy = (sum(r.is_correct for r in results) / len(results)) * 100
    print(f"\nFinal Accuracy: {final_accuracy:.2f}% ({sum(r.is_correct for r in results)}/{len(results)})")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
