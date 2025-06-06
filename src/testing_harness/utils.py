"""Utility functions for the testing harness."""

import logging
import os
from datetime import datetime
from typing import Dict, Any
from logging.handlers import RotatingFileHandler

from .base import QuestionOptions
from .config import LoggingConfig


def setup_logging(config: LoggingConfig, output_dir: str):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        config.format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Clear existing handlers
    logger = logging.getLogger()
    logger.handlers.clear()
    
    # Set log level
    level = getattr(logging, config.level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # File handler
    if config.file_handler:
        log_file = os.path.join(output_dir, f"harness_{datetime.now():%Y%m%d_%H%M%S}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if config.console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def create_prompt(question: str, options: QuestionOptions) -> str:
    """Create a standardized prompt for multiple choice questions."""
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


def calculate_confidence_metrics(results) -> Dict[str, Any]:
    """Calculate confidence-based metrics from results."""
    if not results:
        return {}
    
    # Filter results with confidence scores
    confident_results = [r for r in results if r.model_answer and r.model_answer.confidence is not None]
    
    if not confident_results:
        return {"confidence_analysis": "No confidence scores available"}
    
    confidences = [r.model_answer.confidence for r in confident_results]
    correct_confidences = [r.model_answer.confidence for r in confident_results if r.is_correct]
    
    return {
        "avg_confidence": sum(confidences) / len(confidences),
        "avg_confidence_correct": sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0,
        "high_confidence_count": len([c for c in confidences if c > 0.8]),
        "low_confidence_count": len([c for c in confidences if c < 0.5]),
    }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(remaining_minutes)}m"


def create_summary_report(test_suite, output_dir: str) -> str:
    """Create a human-readable summary report."""
    analytics = test_suite.get_analytics()
    
    report_lines = [
        "=" * 50,
        "TESTING HARNESS SUMMARY REPORT",
        "=" * 50,
        f"Test Suite: {test_suite.name}",
        f"Description: {test_suite.description}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "RESULTS OVERVIEW:",
        f"  Total Questions: {analytics['total_tests']}",
        f"  Correct Answers: {analytics['passed_tests']}",
        f"  Incorrect Answers: {analytics['failed_tests']}",
        f"  Accuracy: {analytics['accuracy']:.2f}%",
        "",
        "PERFORMANCE METRICS:",
        f"  Average Duration: {format_duration(analytics['average_duration'])}",
        f"  Total Duration: {format_duration(analytics['total_duration'])}",
        f"  Error Rate: {analytics['error_rate']:.2f}%",
        f"  Total Retries: {analytics['total_retries']}",
    ]
    
    # Add format breakdown if available
    if "format_breakdown" in test_suite.metadata:
        report_lines.extend([
            "",
            "FORMAT BREAKDOWN:",
        ])
        for fmt, stats in test_suite.metadata["format_breakdown"].items():
            accuracy = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report_lines.append(f"  {fmt}: {stats['passed']}/{stats['total']} ({accuracy:.1f}%)")
    
    # Add model configuration
    if "model_config" in test_suite.metadata:
        model_config = test_suite.metadata["model_config"]
        report_lines.extend([
            "",
            "MODEL CONFIGURATION:",
            f"  Provider: {model_config.get('provider', 'unknown')}",
            f"  Model: {model_config.get('model', 'unknown')}",
            f"  Temperature: {model_config.get('temperature', 'unknown')}",
        ])
    
    report_lines.append("=" * 50)
    
    # Save report
    report_content = "\n".join(report_lines)
    report_file = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    return report_file