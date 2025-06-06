"""Test runners for executing benchmarks."""

import asyncio
import logging
import time
import os
import glob
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn

from .base import (
    BaseTestHarness, LanguageModel, AsyncLanguageModel, 
    EvaluationResult, TestSuite, TestCase
)
from .config import Config
from .parsers import CSVParser, ResponseParser
from .utils import setup_logging, create_prompt


class BenchmarkRunner(BaseTestHarness):
    """Synchronous benchmark runner."""
    
    def __init__(self, model: LanguageModel, config: Config):
        super().__init__(model, config.output_dir, config.to_dict())
        self.config = config
        self.console = Console()
        self.response_parser = ResponseParser(strict=False)
        self.csv_parser = CSVParser()
        setup_logging(config.logging, config.output_dir)
    
    def evaluate_question(self, question_data: Dict[str, Any], question_number: int) -> EvaluationResult:
        """Evaluate a single question with retry logic."""
        return self._evaluate_with_retry(question_data, question_number, 0)
    
    def _evaluate_with_retry(
        self, 
        question_data: Dict[str, Any], 
        question_number: int, 
        retry_count: int
    ) -> EvaluationResult:
        """Evaluate with exponential backoff retry."""
        start_time = time.time()
        options = self.csv_parser.create_question_options(question_data)
        
        result = EvaluationResult(
            question_number=question_number,
            question=question_data["question"],
            options=options,
            metadata={"format": question_data.get("format", "unknown")}
        )
        
        try:
            prompt = create_prompt(question_data["question"], options)
            response = self.model.invoke(prompt)
            
            model_response = self.response_parser.parse(response)
            result.model_answer = model_response
            result.is_correct = model_response.answer == options.correct_letter
            
            logging.debug(f"Q{question_number}: {'✓' if result.is_correct else '✗'} - {model_response.answer}")
            
        except Exception as e:
            if retry_count < self.config.retry.max_retries:
                # Calculate delay with exponential backoff
                delay = self.config.retry.retry_delay
                if self.config.retry.exponential_backoff:
                    delay *= (self.config.retry.backoff_factor ** retry_count)
                
                logging.warning(f"Q{question_number} failed (attempt {retry_count + 1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
                return self._evaluate_with_retry(question_data, question_number, retry_count + 1)
            
            result.error = str(e)
            logging.error(f"Q{question_number} failed after {retry_count + 1} attempts: {e}")
        
        finally:
            result.duration = time.time() - start_time
            result.retries = retry_count
        
        return result
    
    def run(self, data_source: str) -> TestSuite:
        """Run benchmark on data source (file or directory)."""
        logging.info(f"Starting benchmark run on: {data_source}")
        
        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint()
        if checkpoint_data:
            start_index, existing_results = checkpoint_data
            logging.info(f"Resuming from checkpoint at question {start_index + 1}")
        else:
            start_index = 0
            existing_results = []
        
        # Load questions
        questions = self._load_questions(data_source)
        total_questions = len(questions)
        
        if start_index >= total_questions:
            logging.info("All questions already processed")
            return self._create_test_suite(existing_results)
        
        all_results = existing_results[:]
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task(
                "Processing questions",
                total=total_questions
            )
            progress.update(task, completed=start_index)
            
            for i in range(start_index, total_questions):
                question_data = questions[i]
                result = self.evaluate_question(question_data, i + 1)
                all_results.append(result)
                
                progress.update(task, advance=1)
                
                # Save checkpoint periodically
                if (i + 1) % self.config.runner.checkpoint_interval == 0:
                    self.save_checkpoint(all_results, i + 1)
                    logging.info(f"Checkpoint saved at question {i + 1}")
                
                # Rate limiting
                time.sleep(self.config.runner.rate_limit_delay)
        
        # Clear checkpoint on successful completion
        self.clear_checkpoint()
        
        test_suite = self._create_test_suite(all_results)
        
        # Save results
        if self.config.runner.save_partial_results:
            output_file = test_suite.save_results(self.output_dir, self.config.output_format)
            logging.info(f"Results saved to: {output_file}")
        
        return test_suite
    
    def _load_questions(self, data_source: str) -> List[Dict[str, Any]]:
        """Load questions from file or directory."""
        data_path = Path(data_source)
        
        if data_path.is_file():
            # Single file
            if data_path.suffix == '.csv':
                return self.csv_parser.parse_file(str(data_path))
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        elif data_path.is_dir():
            # Directory with CSV files
            csv_files = glob.glob(str(data_path / "*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in {data_path}")
            
            all_questions = []
            for csv_file in sorted(csv_files):
                logging.info(f"Loading questions from: {csv_file}")
                questions = self.csv_parser.parse_file(csv_file)
                all_questions.extend(questions)
            
            return all_questions
        
        else:
            raise ValueError(f"Data source not found: {data_source}")
    
    def _create_test_suite(self, results: List[EvaluationResult]) -> TestSuite:
        """Create test suite from results."""
        test_cases = [TestCase.from_evaluation_result(result) for result in results]
        
        return TestSuite(
            name="Benchmark Evaluation",
            description="LLM evaluation on multiple choice questions",
            test_cases=test_cases,
            metadata={
                "model_config": self.model.get_config(),
                "run_config": self.config.to_dict(),
                **self._calculate_analytics(test_cases)
            }
        )
    
    def _calculate_analytics(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Calculate detailed analytics."""
        total = len(test_cases)
        passed = sum(1 for tc in test_cases if tc.passed)
        
        # Calculate by format if available
        format_stats = {}
        for tc in test_cases:
            fmt = tc.metadata.get("format", "unknown")
            if fmt not in format_stats:
                format_stats[fmt] = {"total": 0, "passed": 0}
            format_stats[fmt]["total"] += 1
            if tc.passed:
                format_stats[fmt]["passed"] += 1
        
        return {
            "timestamp": time.time(),
            "total_questions": total,
            "passed_tests": passed,
            "accuracy": (passed / total * 100) if total > 0 else 0,
            "format_breakdown": format_stats,
        }


class AsyncBenchmarkRunner:
    """Asynchronous benchmark runner for improved performance."""
    
    def __init__(self, model: AsyncLanguageModel, config: Config):
        self.model = model
        self.config = config
        self.console = Console()
        self.response_parser = ResponseParser(strict=False)
        self.csv_parser = CSVParser()
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        setup_logging(config.logging, config.output_dir)
    
    async def evaluate_question(self, question_data: Dict[str, Any], question_number: int) -> EvaluationResult:
        """Evaluate a single question asynchronously."""
        return await self._evaluate_with_retry(question_data, question_number, 0)
    
    async def _evaluate_with_retry(
        self,
        question_data: Dict[str, Any],
        question_number: int,
        retry_count: int
    ) -> EvaluationResult:
        """Async evaluate with retry logic."""
        start_time = time.time()
        options = self.csv_parser.create_question_options(question_data)
        
        result = EvaluationResult(
            question_number=question_number,
            question=question_data["question"],
            options=options,
            metadata={"format": question_data.get("format", "unknown")}
        )
        
        try:
            prompt = create_prompt(question_data["question"], options)
            response = await self.model.ainvoke(prompt)
            
            model_response = self.response_parser.parse(response)
            result.model_answer = model_response
            result.is_correct = model_response.answer == options.correct_letter
            
        except Exception as e:
            if retry_count < self.config.retry.max_retries:
                delay = self.config.retry.retry_delay
                if self.config.retry.exponential_backoff:
                    delay *= (self.config.retry.backoff_factor ** retry_count)
                
                await asyncio.sleep(delay)
                return await self._evaluate_with_retry(question_data, question_number, retry_count + 1)
            
            result.error = str(e)
            logging.error(f"Q{question_number} failed: {e}")
        
        finally:
            result.duration = time.time() - start_time
            result.retries = retry_count
        
        return result
    
    async def process_batch(self, questions_batch: List[Tuple[Dict[str, Any], int]]) -> List[EvaluationResult]:
        """Process a batch of questions concurrently."""
        tasks = [
            self.evaluate_question(question_data, question_number)
            for question_data, question_number in questions_batch
        ]
        return await asyncio.gather(*tasks)
    
    async def run(self, data_source: str) -> TestSuite:
        """Run async benchmark."""
        logging.info(f"Starting async benchmark run on: {data_source}")
        
        # Load questions (same as sync version)
        questions = self._load_questions(data_source)
        total_questions = len(questions)
        
        all_results = []
        batch_size = self.config.runner.batch_size
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Processing questions", total=total_questions)
            
            for i in range(0, total_questions, batch_size):
                batch = questions[i:i + batch_size]
                batch_with_numbers = [(q, i + j + 1) for j, q in enumerate(batch)]
                
                results = await self.process_batch(batch_with_numbers)
                all_results.extend(results)
                
                progress.update(task, advance=len(batch))
                
                # Rate limiting between batches
                if i + batch_size < total_questions:
                    await asyncio.sleep(self.config.runner.rate_limit_delay)
        
        test_suite = self._create_test_suite(all_results)
        
        # Save results
        if self.config.runner.save_partial_results:
            output_file = test_suite.save_results(self.output_dir, self.config.output_format)
            logging.info(f"Results saved to: {output_file}")
        
        await self.model.aclose()
        return test_suite
    
    def _load_questions(self, data_source: str) -> List[Dict[str, Any]]:
        """Load questions (same implementation as sync version)."""
        data_path = Path(data_source)
        
        if data_path.is_file():
            if data_path.suffix == '.csv':
                return self.csv_parser.parse_file(str(data_path))
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        elif data_path.is_dir():
            csv_files = glob.glob(str(data_path / "*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in {data_path}")
            
            all_questions = []
            for csv_file in sorted(csv_files):
                questions = self.csv_parser.parse_file(csv_file)
                all_questions.extend(questions)
            
            return all_questions
        
        else:
            raise ValueError(f"Data source not found: {data_source}")
    
    def _create_test_suite(self, results: List[EvaluationResult]) -> TestSuite:
        """Create test suite from results."""
        test_cases = [TestCase.from_evaluation_result(result) for result in results]
        
        return TestSuite(
            name="Async Benchmark Evaluation",
            description="Async LLM evaluation on multiple choice questions",
            test_cases=test_cases,
            metadata={
                "model_config": self.model.get_config(),
                "run_config": self.config.to_dict(),
            }
        )