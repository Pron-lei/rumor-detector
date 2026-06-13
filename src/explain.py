"""Public explanation interface for rumor detection inference."""

from __future__ import annotations

import re
from numbers import Real

try:
    from .llm_api import LLMAPIError, call_sjtu_llm
    from .prompts import build_prompt
except ImportError:  # Allows running `python src/explain.py` directly.
    from llm_api import LLMAPIError, call_sjtu_llm
    from prompts import build_prompt


MAX_TEXT_CHARS = 800


def _clean_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    cleaned = "" if text is None else str(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "..."
    return cleaned


def _validate_label(label: int) -> int:
    try:
        value = int(label)
    except (TypeError, ValueError):
        value = 0
    return value if value in (0, 1) else 0


def _validate_confidence(confidence: float) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 0.0
    return min(max(value, 0.0), 1.0)


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


def _postprocess_explanation(explanation: str) -> str:
    text = _strip_llm_prefix(explanation)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(解释|判断依据|答案)[:：]\s*", "", text)
    return text.strip(" \t\r\n\"'")


def _contains_url(text: str) -> bool:
    return bool(re.search(r"https?://|www\.", text, flags=re.IGNORECASE))


def _has_all_caps_signal(text: str) -> bool:
    words = re.findall(r"\b[A-Z]{3,}\b", text)
    return len(words) >= 2


def _append_unique(tags: list[str], tag: str) -> None:
    if tag not in tags:
        tags.append(tag)


def extract_evidence_tags(text: str) -> list[str]:
    tags: list[str] = []
    lowered = text.lower()

    keyword_rules = [
        (["breaking", "#breaking", "breaking news", "urgent", "developing"], "突发新闻式表达"),
        (["someone said", "people say", "it is said", "anonymous source", "no source", "they say"], "来源不明确"),
        (
            ["rumor", "rumour", "unconfirmed", "allegedly", "reportedly", "claims", "some say", "could be", "might be"],
            "未经证实或传闻式表达",
        ),
        (
            ["share", "retweet", "spread", "forward", "send this to everyone", "make this viral", "before it is deleted"],
            "诱导转发或扩散",
        ),
        (["shocking", "terrifying", "horrible", "disgusting", "panic", "fear", "outrage", "unbelievable"], "情绪化表达"),
        (["always", "never", "definitely", "100%", "guaranteed", "must", "everyone", "nobody", "proven"], "绝对化判断"),
        (
            ["secret", "cover-up", "hidden truth", "they are hiding", "the media won't tell you", "government doesn't want you to know", "exposed"],
            "阴谋化叙事",
        ),
    ]

    for keywords, tag in keyword_rules:
        if any(keyword in lowered for keyword in keywords):
            _append_unique(tags, tag)

    if text.count("!") >= 2:
        _append_unique(tags, "情绪化表达")

    letters = re.findall(r"[A-Za-z]", text)
    uppercase = re.findall(r"[A-Z]", text)
    if letters and len(uppercase) / len(letters) > 0.35:
        _append_unique(tags, "情绪化表达")

    words = re.findall(r"\b\w+\b", text)
    if len(words) < 12 and not _contains_url(text) and not re.search(r"\d", text):
        _append_unique(tags, "缺少可核验细节")

    if not tags:
        tags.append("表述较客观")
    return tags


def _confidence_phrase(confidence: float) -> str:
    if confidence >= 0.80:
        return "模型置信度较高"
    if confidence >= 0.60:
        return "模型给出了倾向性判断"
    return "模型判断不够确定"


def _fallback_explanation(
    text: str,
    label: int,
    confidence: float,
    evidence_tags: list[str] | None = None,
) -> str:
    tags = evidence_tags or extract_evidence_tags(text)
    phrase = _confidence_phrase(confidence)
    tag_text = "、".join(tags[:4])

    if label == 1:
        return (
            f"该推文被判定为谣言，{phrase}。从文本内部特征看，文本包含{tag_text}等风险线索，"
            "但该解释仅说明分类模型可能依据的表达模式，不代表已经完成事实核查。"
        )

    if "表述较客观" in tags:
        detail = "文本整体表述较客观，没有明显夸张、恐慌、诱导转发或阴谋化叙事"
    else:
        detail = f"虽然文本中可见{tag_text}，但整体缺少足以支持谣言判断的明显高风险表达"
    return (
        f"该推文被判定为非谣言，{phrase}。{detail}，因此模型认为其谣言风险较低。"
        "该解释不代表已经查证新闻真伪。"
    )


def generate_explanation(
    text: str,
    label: int,
    confidence: float,
    prompt_type: str = "fewshot_evidence",
    use_llm: bool = True,
) -> str:
    """Generate a Chinese explanation for a predicted rumor label.

    This function is designed for direct use by `src/inference.py`.
    """

    clean_text = _clean_text(text)
    valid_label = _validate_label(label)
    valid_confidence = _validate_confidence(confidence)
    evidence_tags = extract_evidence_tags(clean_text)

    if isinstance(prompt_type, bool):
        use_llm = prompt_type
        prompt_type = "fewshot_evidence"

    if not clean_text:
        return _fallback_explanation("空文本", valid_label, valid_confidence, evidence_tags)

    if use_llm:
        prompt = build_prompt(
            clean_text,
            valid_label,
            valid_confidence,
            prompt_type=prompt_type,
            evidence_tags=evidence_tags,
        )
        try:
            output = call_sjtu_llm(prompt)
            explanation = _postprocess_explanation(output)
            if len(explanation) >= 20:
                return explanation
        except LLMAPIError:
            pass

    return _fallback_explanation(clean_text, valid_label, valid_confidence, evidence_tags)


if __name__ == "__main__":
    sample_text = (
        "BREAKING!!! Everyone must share this now, officials are hiding the truth "
        "and the cure will disappear tomorrow!"
    )
    print(generate_explanation(sample_text, label=1, confidence=0.87))
