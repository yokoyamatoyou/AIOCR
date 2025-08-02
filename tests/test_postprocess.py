import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core import postprocess


def test_validation_rule_failure():
    text = "12345O7"  # O instead of 0
    normalized, needs_human = postprocess.postprocess_result(text, 0.99, "regex:^\\d{7}$")
    assert normalized == "12345O7".replace(" ", "")
    assert needs_human


def test_low_confidence_flag():
    text = "1234567"
    normalized, needs_human = postprocess.postprocess_result(text, 0.5, "regex:^\\d{7}$")
    assert normalized == "1234567"
    assert needs_human
