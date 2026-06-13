"""Public explanation interface for rumor detection inference."""

from __future__ import annotations

import re
from numbers import Real

try:
    from .llm_api import LLMAPIError, call_llm
    from .prompts import build_prompt
except ImportError:  # Allows running `python src/explain.py` directly.
    from llm_api import LLMAPIError, call_llm
    from prompts import build_prompt


MAX_TEXT_CHARS = 1200


def _clean_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "..."
    return cleaned


def _validate_label(label: int) -> int:
    if isinstance(label, bool) or not isinstance(label, int):
        raise TypeError("label must be an integer: 0 for 非谣言, 1 for 谣言")
    if label not in (0, 1):
        raise ValueError("label must be 0 (非谣言) or 1 (谣言)")
    return label


def _validate_confidence(confidence: float) -> float:
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise TypeError("confidence must be a number between 0 and 1")
    value = float(confidence)
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return value


def _strip_llm_prefix(output: str) -> str:
    text = re.sub(r"\s+", " ", output).strip()
    prefix_patterns = [
        r"^判断依据[:：]\s*",
        r"^解释[:：]\s*",
        r"^中文判断依据[:：]\s*",
        r"^答案[:：]\s*",
        r"^该文本的判断依据是[:：]?\s*",
    ]
    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip(" \t\r\n\"'")


def _contains_url(text: str) -> bool:
    return bool(re.search(r"https?://|www\.", text, flags=re.IGNORECASE))


def _has_all_caps_signal(text: str) -> bool:
    words = re.findall(r"\b[A-Z]{3,}\b", text)
    return len(words) >= 2


def _fallback_explanation(text: str, label: int, confidence: float) -> str:
    features: list[str] = []
    lowered = text.lower()

    if re.search(r"!{2,}|\?{2,}", text):
        features.append("使用连续感叹号或问号，情绪色彩较强")
    if _has_all_caps_signal(text):
        features.append("存在较多全大写词，表达上有强调和煽动倾向")
    if any(word in lowered for word in ("share", "retweet", "forward", "must read", "urgent", "breaking")):
        features.append("包含诱导传播或突发强调类措辞")
    if any(word in lowered for word in ("everyone", "always", "never", "secret", "hidden", "cure", "shocking")):
        features.append("出现绝对化或神秘化表达")
    if not _contains_url(text) and not re.search(r"\b(said|according to|reported by|official|agency)\b", lowered):
        features.append("缺少清晰来源或可核对出处")

    if not features:
        if label == 1:
            features.append("文本细节和来源线索不足，整体更像未经充分说明的断言")
        else:
            features.append("文本表述相对克制，没有明显诱导转发、夸张断言或强烈情绪化特征")

    label_text = "谣言" if label == 1 else "非谣言"
    feature_text = "；".join(features[:3])
    return (
        f"模型将该文本判为{label_text}，置信度约为{confidence:.2%}。"
        f"从文本内部特征看，{feature_text}。该解释仅基于文本表达模式，不能替代事实核验。"
    )


def generate_explanation(
    text: str,
    label: int,
    confidence: float,
    use_llm: bool = True,
) -> str:
    """Generate a Chinese explanation for a predicted rumor label.

    This function is designed for direct use by `src/inference.py`.
    """

    clean_text = _clean_text(text)
    valid_label = _validate_label(label)
    valid_confidence = _validate_confidence(confidence)

    if not clean_text:
        return _fallback_explanation("空文本", valid_label, valid_confidence)

    if use_llm:
        messages = build_prompt(clean_text, valid_label, valid_confidence, prompt_type="evidence")
        try:
            output = call_llm(messages)
            explanation = _strip_llm_prefix(output)
            if explanation:
                return explanation
        except LLMAPIError:
            pass

    return _fallback_explanation(clean_text, valid_label, valid_confidence)


if __name__ == "__main__":
    sample_text = (
        "BREAKING!!! Everyone must share this now, officials are hiding the truth "
        "and the cure will disappear tomorrow!"
    )
    print(generate_explanation(sample_text, label=1, confidence=0.87))
