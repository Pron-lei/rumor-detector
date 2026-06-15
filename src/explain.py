"""
解释生成模块
- 接收分类结果，调用 LLM 生成中文判断依据
- 支持 LLM 失败时回退到规则解释
- API Key 已内置默认值，无需 .env 文件即可使用
"""
import sys
import os
import re

sys.path.insert(0, os.path.dirname(__file__))

from llm_api import LLMAPIError, call_llm
from prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_BUILD_USER_PROMPT,
    build_user_prompt_v0,
    build_user_prompt_v1,
    build_user_prompt_fewshot,
    build_user_prompt_rag,
    SYSTEM_PROMPT_V2,
)


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


def extract_evidence_tags(text: str) -> "list[str]":
    """从文本中提取证据标签，供规则回退使用。"""
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
            if tag not in tags:
                tags.append(tag)

    if text.count("!") >= 2:
        if "情绪化表达" not in tags:
            tags.append("情绪化表达")

    letters = re.findall(r"[A-Za-z]", text)
    uppercase = re.findall(r"[A-Z]", text)
    if letters and len(uppercase) / len(letters) > 0.35:
        if "情绪化表达" not in tags:
            tags.append("情绪化表达")

    words = re.findall(r"\b\w+\b", text)
    if len(words) < 12 and not re.search(r"https?://|www\.", text, flags=re.IGNORECASE) and not re.search(r"\d", text):
        if "缺少可核验细节" not in tags:
            tags.append("缺少可核验细节")

    if not tags:
        tags.append("表述较客观")
    return tags


def _confidence_phrase(confidence: float) -> str:
    if confidence >= 0.80:
        return "模型置信度较高"
    if confidence >= 0.60:
        return "模型给出了倾向性判断"
    return "模型判断不够确定"


def _fallback_explanation(text: str, label: int, confidence: float) -> str:
    """规则回退：当 LLM 不可用时，基于文本特征生成解释"""
    label_text = "谣言" if label == 1 else "非谣言"

    # 谣言常用词汇/模式
    rumor_patterns = [
        ("breaking", "使用了'BREAKING'强调紧迫感，但缺乏官方来源"),
        ("shocking", "使用了'shocking'等情绪化词汇"),
        ("anonymous", "信息来源为匿名，无法核实"),
        ("unconfirmed", "明确标注未确认"),
        ("!!!", "使用了大量感叹号，情绪化表达较为明显"),
        ("just in", "使用'JUST IN'营造新闻感，但未提供可验证的引用"),
        ("#breaking", "使用了#breaking话题标签但缺乏新闻机构背书"),
    ]

    non_rumor_patterns = [
        ("according to", "引用了可验证的消息来源"),
        ("official", "涉及官方信息"),
        ("confirmed", "信息已被确认"),
        ("police said", "引用了警方的官方声明"),
        ("report", "提及具体报告或文件"),
    ]

    patterns = rumor_patterns if label == 1 else non_rumor_patterns
    matched = []

    text_lower = text.lower()
    for keyword, reason in patterns:
        if keyword in text_lower:
            matched.append(reason)

    if matched:
        evidence = "；".join(matched[:2])
        return f"该推文被判定为{label_text}（置信度{confidence:.1%}）。主要依据：{evidence}。"
    else:
        return f"该推文被判定为{label_text}（置信度{confidence:.1%}）。模型基于推文的整体语言特征和上下文模式做出判断。"


def generate_explanation(
    text: str,
    label: int,
    confidence: float,
    prompt_version: str = "v1",
    model: str = "deepseek-chat",
    use_llm: bool = True,
    retrieved: "list[dict] | None" = None,
) -> str:
    """
    生成判断依据解释

    Args:
        text: 推文文本
        label: 分类标签 (0=非谣言, 1=谣言)
        confidence: 分类置信度 (0-1)
        prompt_version: 使用的 Prompt 版本
            - "v0": 基础直接询问
            - "v1": 思维链引导（默认）
            - "fewshot": 带示例
            - "rag": RAG 检索增强（需要 retrieved 参数）
        model: 使用的 LLM 模型
        use_llm: 是否调用 LLM（False 时直接走规则回退）
        retrieved: RAG 检索到的相似案例列表（prompt_version="rag" 时使用）

    Returns:
        中文解释文本
    """
    clean_text = _clean_text(text)
    valid_label = _validate_label(label)
    valid_confidence = _validate_confidence(confidence)

    # 兼容旧调用方式：prompt_version 可能被传为 "fewshot_evidence" 等旧名称
    version_map = {
        "basic": "v0",
        "evidence": "v1",
        "fewshot_evidence": "v1",
        "fewshot": "fewshot",
        "v0": "v0",
        "v1": "v1",
        "rag": "rag",
    }
    version = version_map.get(prompt_version, "v1")

    # 兼容旧接口：bool 值直接控制 use_llm
    if isinstance(use_llm, bool) and not use_llm:
        return _fallback_explanation(clean_text, valid_label, valid_confidence)

    if not clean_text:
        return _fallback_explanation("空文本", valid_label, valid_confidence)

    # 选择 system prompt
    if version in ("v0", "v1", "rag"):
        system_prompt = SYSTEM_PROMPT_V2
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # 选择 user prompt 构建函数
    if version == "rag" and retrieved:
        user_prompt = build_user_prompt_rag(clean_text, valid_label, valid_confidence, retrieved)
    else:
        builders = {
            "v0": build_user_prompt_v0,
            "v1": build_user_prompt_v1,
            "fewshot": build_user_prompt_fewshot,
        }
        build_fn = builders.get(version, DEFAULT_BUILD_USER_PROMPT)
        user_prompt = build_fn(clean_text, valid_label, valid_confidence)

    try:
        explanation = call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.3,
            max_tokens=256,
            timeout=30,
        )
        explanation = explanation.strip()
        if explanation and len(explanation) >= 10:
            return explanation
    except LLMAPIError as e:
        print(f"[explain] LLM 调用失败: {e}")
    except Exception as e:
        print(f"[explain] LLM 调用异常: {e}")

    # 回退：规则模板
    return _fallback_explanation(clean_text, valid_label, valid_confidence)


def generate_explanation_batch(
    texts: "list[str]",
    labels: "list[int]",
    confidences: "list[float]",
    prompt_version: str = "v1",
    model: str = "deepseek-chat",
    delay: float = 0.5,
) -> "list[str]":
    """
    批量生成解释，带请求间隔防止频率限制
    """
    import time
    explanations = []
    for i, (t, lbl, conf) in enumerate(zip(texts, labels, confidences)):
        exp = generate_explanation(t, lbl, conf, prompt_version, model)
        explanations.append(exp)
        if (i + 1) % 10 == 0:
            print(f"[explain] 进度: {i + 1}/{len(texts)}")
        if i < len(texts) - 1:
            time.sleep(delay)
    return explanations


if __name__ == "__main__":
    # 测试：生成一条解释
    test_text = "BREAKING: Anonymous sources report that a famous celebrity has been found dead in their home. No official statement yet."
    test_label = 1
    test_conf = 0.89

    print(f"推文: {test_text}")
    print(f"标签: {'谣言' if test_label == 1 else '非谣言'} (置信度 {test_conf:.0%})")
    print()
    print("生成解释中...")
    exp = generate_explanation(test_text, test_label, test_conf)
    print(f"解释: {exp}")
