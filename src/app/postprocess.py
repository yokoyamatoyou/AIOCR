import re
from typing import Optional, Tuple

CONF_THRESHOLD = 0.9

FULLWIDTH_MAP = str.maketrans(
    "０１２３４５６７８９－",
    "0123456789-"
)


def normalize_text(text: str) -> str:
    """簡易的なテキスト正規化"""
    if text is None:
        return ""
    text = text.strip()
    text = text.translate(FULLWIDTH_MAP)
    text = text.replace(" ", "").replace("\u3000", "")
    return text


def check_validation(text: str, rule: Optional[str]) -> bool:
    if not rule:
        return True
    if rule.startswith("regex:"):
        pattern = rule[len("regex:") :]
        return re.fullmatch(pattern, text) is not None
    return True


def postprocess_result(
    text: str, confidence: float, rule: Optional[str]
) -> Tuple[str, bool]:
    norm_text = normalize_text(text)
    valid = check_validation(norm_text, rule)
    needs_human = confidence < CONF_THRESHOLD or not valid
    return norm_text, needs_human
