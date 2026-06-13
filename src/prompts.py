"""Prompt templates for Chinese rumor-detection explanations."""

from __future__ import annotations

from typing import Literal


PromptType = Literal["basic", "evidence", "fewshot"]

LABEL_TEXT = {
    0: "非谣言",
    1: "谣言",
}

SYSTEM_PROMPT = (
    "你是一个谨慎的可解释谣言检测助手。你只能基于输入文本本身和给定分类结果生成中文判断依据。"
    "不得编造外部事实，不得声称你已经查证新闻真伪，不得引用未提供的来源。"
)


def _format_confidence(confidence: float) -> str:
    return f"{confidence:.2%}"


def _common_context(text: str, label: int, confidence: float) -> str:
    return (
        f"待解释文本：{text}\n"
        f"分类结果：{LABEL_TEXT[label]}（label={label}）\n"
        f"模型置信度：{_format_confidence(confidence)}"
    )


def _basic_prompt(text: str, label: int, confidence: float) -> list[dict[str, str]]:
    user_prompt = (
        f"{_common_context(text, label, confidence)}\n\n"
        "请用中文生成一段简洁的判断依据，说明为什么该文本被判为上述类别。"
        "解释必须聚焦文本内部特征，例如措辞是否夸张、来源是否明确、是否诱导转发、"
        "是否使用绝对化表达或情绪化表达。不要输出项目符号。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _evidence_prompt(text: str, label: int, confidence: float) -> list[dict[str, str]]:
    user_prompt = (
        f"{_common_context(text, label, confidence)}\n\n"
        "请生成中文判断依据，要求：\n"
        "1. 先概括分类结论，但不要声称已经核验事实真伪；\n"
        "2. 引用或转述文本中的具体线索作为依据；\n"
        "3. 重点分析文本内部特征，包括夸张表达、缺少明确来源、诱导转发、绝对化措辞、"
        "情绪煽动、细节不足或表述相对克制等；\n"
        "4. 只输出一段 80 到 160 字的中文解释。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _fewshot_prompt(text: str, label: int, confidence: float) -> list[dict[str, str]]:
    examples = (
        "示例一：\n"
        "文本：BREAKING!!! Everyone must share this now, doctors are hiding the cure!\n"
        "分类结果：谣言\n"
        "解释：该文本使用大量感叹号和“must share”等强烈转发动员，并声称有人隐瞒关键信息，"
        "但没有给出明确来源或可核对细节，因此从文本特征看更符合谣言传播的表达模式。\n\n"
        "示例二：\n"
        "文本：The city health office said the vaccination clinic will open at 9 a.m. on Monday.\n"
        "分类结果：非谣言\n"
        "解释：该文本语气较为克制，信息包含机构、时间和具体事项，没有明显煽动性措辞、"
        "绝对化断言或诱导转发，因此从文本内部特征看更接近普通信息陈述。"
    )
    user_prompt = (
        f"{examples}\n\n"
        f"{_common_context(text, label, confidence)}\n\n"
        "请参考示例风格，输出一段中文判断依据。不得编造外部事实，不得说已经查证新闻真伪，"
        "只能分析文本内部线索。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_prompt(
    text: str,
    label: int,
    confidence: float,
    prompt_type: PromptType = "evidence",
) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages for the selected prompt template."""

    if label not in LABEL_TEXT:
        raise ValueError("label must be 0 (非谣言) or 1 (谣言)")
    if prompt_type == "basic":
        return _basic_prompt(text, label, confidence)
    if prompt_type == "evidence":
        return _evidence_prompt(text, label, confidence)
    if prompt_type == "fewshot":
        return _fewshot_prompt(text, label, confidence)
    raise ValueError("prompt_type must be one of: basic, evidence, fewshot")


__all__ = ["build_prompt"]
