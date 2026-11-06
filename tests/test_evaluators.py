"""Evaluation metric tests using DeepEval and golden evaluation dataset.

Loads evaluation test cases from eval_golden_dataset.json, validates test case schemas,
and evaluates LLM faithfulness and hallucination metrics against expected outputs.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

DATASET_PATH = Path(__file__).parent / "eval_golden_dataset.json"


def load_golden_dataset() -> list[dict[str, Any]]:
    """Load and parse the golden evaluation dataset from JSON.

    Returns:
        list[dict[str, Any]]: List of test case dictionaries.
    """
    assert DATASET_PATH.exists(), f"Golden dataset not found at {DATASET_PATH}"
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    return data


def test_golden_dataset_structure() -> None:
    """Test that the golden evaluation dataset contains at least 20 valid test cases."""
    dataset = load_golden_dataset()
    assert len(dataset) >= 20, f"Expected at least 20 test cases, found {len(dataset)}"

    for idx, item in enumerate(dataset):
        assert "input" in item, f"Missing 'input' key at index {idx}"
        assert "expected_output" in item, f"Missing 'expected_output' key at index {idx}"
        assert "retrieved_context" in item, f"Missing 'retrieved_context' key at index {idx}"
        assert isinstance(item["input"], str) and len(item["input"]) > 0
        assert isinstance(item["expected_output"], str) and len(item["expected_output"]) > 0
        assert isinstance(item["retrieved_context"], list) and len(item["retrieved_context"]) > 0


@pytest.mark.eval
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("DEEPEVAL_TEST_ENABLED"),
    reason="DeepEval metric tests require live LLM key or DEEPEVAL_TEST_ENABLED=1",
)
def test_faithfulness_metric_live() -> None:
    """Test faithfulness evaluation against the golden dataset using DeepEval."""
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    dataset = load_golden_dataset()
    # Test on a representative subset to manage CI time
    subset = dataset[:5]

    metric = FaithfulnessMetric(threshold=0.85)

    for item in subset:
        test_case = LLMTestCase(
            input=item["input"],
            actual_output=item["expected_output"],
            retrieved_context=item["retrieved_context"],
        )
        metric.measure(test_case)
        assert metric.score >= 0.85, (
            f"Faithfulness score {metric.score} below threshold 0.85 for input: {item['input']}"
        )


@pytest.mark.eval
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("DEEPEVAL_TEST_ENABLED"),
    reason="DeepEval metric tests require live LLM key or DEEPEVAL_TEST_ENABLED=1",
)
def test_hallucination_metric_live() -> None:
    """Test hallucination evaluation against the golden dataset using DeepEval."""
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase

    dataset = load_golden_dataset()
    subset = dataset[:5]

    metric = HallucinationMetric(threshold=0.15)

    for item in subset:
        test_case = LLMTestCase(
            input=item["input"],
            actual_output=item["expected_output"],
            context=item["retrieved_context"],
        )
        metric.measure(test_case)
        assert metric.score <= 0.15, (
            f"Hallucination score {metric.score} exceeded threshold 0.15 for input: {item['input']}"
        )


def test_mock_evaluator_scoring() -> None:
    """Deterministic offline evaluator test verifying score aggregation logic."""
    dataset = load_golden_dataset()
    scores = []

    for item in dataset:
        # Check keyword overlap between expected_output and retrieved_context
        expected_words = set(item["expected_output"].lower().split())
        context_words = set(" ".join(item["retrieved_context"]).lower().split())
        overlap = len(expected_words.intersection(context_words)) / max(len(expected_words), 1)
        scores.append(overlap)

    avg_overlap = sum(scores) / len(scores)
    assert avg_overlap > 0.40, f"Average ground truth context overlap was {avg_overlap}"
