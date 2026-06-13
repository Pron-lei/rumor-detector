"""Prompt templates for Chinese rumor-detection explanations."""

from __future__ import annotations

from typing import Literal


PromptType = Literal["basic", "evidence", "fewshot", "fewshot_evidence"]

LABEL_MAP = {
    0: "非谣言",
    1: "谣言",
}

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


def build_fewshot_evidence_prompt(
    text: str,
    label: int,
    confidence: float,
    evidence_tags: list[str] | None = None,
) -> str:
    """Build a few-shot prompt with explicit evidence tags."""

    label_text = LABEL_MAP.get(label, "非谣言")
    tags = evidence_tags or ["表述较客观"]
    tag_text = "、".join(tags)
    confidence_note = (
        "如果 confidence < 0.60，要在解释中体现模型判断不够确定。"
        if confidence < 0.60
        else ""
    )
    return (
        "你是一个可解释谣言检测助手。LLM 不是事实核查系统，不要重新判断真假，"
        "只解释分类模型为什么可能给出该结果。不要编造外部事实、新闻背景、调查结论。"
        "不要声称“已经查证”“事实证明”“官方证实”。不要仅因为事件名、人物名、地名、"
        "hashtag、URL 判断为谣言或非谣言。只输出一段 80 到 150 字中文判断依据，"
        "不要分点，不要输出英文原文。"
        f"{confidence_note}\n\n"
        "示例 1：\n"
        "推文：BREAKING: Unconfirmed reports say a second explosion happened downtown.\n"
        "标签：1（谣言）\n"
        "置信度：0.8600\n"
        "线索：突发新闻式表达、未经证实或传闻式表达\n"
        "解释：该推文被判定为谣言，模型置信度较高。文本中出现了突发新闻式表达和未经证实的说法，"
        "但没有给出明确来源或更多可核验细节，因此模型认为其谣言风险较高。\n\n"
        "示例 2：\n"
        "推文：The city council will hold a public meeting on Friday according to the official schedule.\n"
        "标签：0（非谣言）\n"
        "置信度：0.8200\n"
        "线索：表述较客观\n"
        "解释：该推文被判定为非谣言，模型置信度较高。文本整体表述较为客观，包含明确主体和时间信息，"
        "没有明显夸张、恐慌或诱导转发的表达，因此模型认为其谣言风险较低。\n\n"
        "现在请解释下面这条推文的分类结果：\n"
        f"推文：{text}\n"
        f"标签：{label}（{label_text}）\n"
        f"置信度：{confidence:.4f}\n"
        f"线索：{tag_text}\n"
        "解释："
    )


def build_prompt(
    text: str,
    label: int,
    confidence: float,
    prompt_type: PromptType = "fewshot_evidence",
    evidence_tags: list[str] | None = None,
) -> list[dict[str, str]] | str:
    """Build OpenAI-compatible chat messages for the selected prompt template."""

    if label not in LABEL_TEXT:
        raise ValueError("label must be 0 (非谣言) or 1 (谣言)")
    if prompt_type == "fewshot_evidence":
        return build_fewshot_evidence_prompt(text, label, confidence, evidence_tags)
    if prompt_type == "basic":
        return _basic_prompt(text, label, confidence)
    if prompt_type == "evidence":
        return _evidence_prompt(text, label, confidence)
    if prompt_type == "fewshot":
        return _fewshot_prompt(text, label, confidence)
    raise ValueError("prompt_type must be one of: basic, evidence, fewshot, fewshot_evidence")


__all__ = ["LABEL_MAP", "build_fewshot_evidence_prompt", "build_prompt"]
