"""
Prompt 模板管理
- 用于谣言检测的中文解释生成
- 提供零样本、少样本、思维链等多种策略
"""

# ═══════════════════════════════════════════════════════════════
# System Prompt — 设定 LLM 角色
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_V1 = """你是一个专业的谣言检测分析助手。你的任务是根据给定的推文内容和分类模型给出的检测结果，用中文解释判断依据。

要求：
1. 必须引用推文中的具体词语、句式或细节作为证据
2. 解释要有逻辑性，清晰说明为什么判定为谣言或非谣言
3. 控制在2-3句话，直接、客观
4. 不要编造推文中不存在的信息
5. 不要输出与解释无关的内容"""

SYSTEM_PROMPT_V2 = """你是一名社交媒体谣言分析专家。请你结合推文内容，以第二人称（"该推文"）客观地分析：为什么这条推文被判定为谣言或非谣言。

分析时请关注以下维度（选择最相关的1-2点展开）：
- 语言特征：是否使用夸张词汇（BREAKING、shocking）、情绪化表达、大量感叹号
- 信息来源：是否引用官方/权威来源，还是匿名/无法验证的消息
- 内容逻辑：叙述是否存在矛盾，是否包含无法核实的具体数字
- 写作风格：是否类似新闻标题（标题党），是否大量使用hashtag
- 传播意图：是在陈述事实还是在煽动情绪、传播恐慌

请用中文给出2-3句分析，直接引用推文原文作为证据。"""


# ═══════════════════════════════════════════════════════════════
# Few-shot 示例
# ═══════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    {
        "text": "BREAKING: Massive earthquake hits downtown, thousands feared dead!",
        "label": 1,
        "confidence": 0.92,
        "explanation": "该推文使用'BREAKING'营造紧迫感，'Massive'和'thousands feared dead'属于夸张表述但缺乏具体地点、时间和消息来源，符合谣言的传播特征——用模糊的灾难描述煽动恐慌情绪。"
    },
    {
        "text": "The city council will meet on Tuesday to discuss the budget proposal.",
        "label": 0,
        "confidence": 0.88,
        "explanation": "该推文陈述了一件常规的市政会议安排，时间（Tuesday）、主体（city council）、事项（budget proposal）明确且可验证，语气平实、无情绪化措辞，属于正常的信息传达。"
    },
    {
        "text": "Anonymous sources confirm that a famous actor has been found dead. Police investigating.",
        "label": 1,
        "confidence": 0.85,
        "explanation": "该推文的信息来源仅为'Anonymous sources'（匿名来源），无法核实；'famous actor'未指明具体是谁，'Police investigating'看似增加可信度但缺乏具体警方声明链接。匿名来源+模糊主体是典型谣言模式。"
    },
]


# ═══════════════════════════════════════════════════════════════
# User Prompt 构建函数
# ═══════════════════════════════════════════════════════════════

def build_user_prompt_v0(text: str, label: int, confidence: float) -> str:
    """基础版：直接询问"""
    label_text = "谣言" if label == 1 else "非谣言"
    return f"""推文内容："{text}"

检测结果：该推文被判定为【{label_text}】，置信度 {confidence:.2%}。

请解释为什么这条推文是{label_text}："""


def build_user_prompt_v1(text: str, label: int, confidence: float) -> str:
    """增强版：加入思维链引导"""
    label_text = "谣言" if label == 1 else "非谣言"

    if label == 1:
        hint = """请按以下步骤分析（在内心完成，只输出最终解释）：
1) 识别推文中可能暗示不实信息的线索（如来源不明、用词夸张、缺乏证据）
2) 指出最关键的1-2个可疑点
3) 用中文简明解释"""
    else:
        hint = """请按以下步骤分析（在内心完成，只输出最终解释）：
1) 识别推文中体现真实性的特征（如来源可靠、信息具体可验证、语气客观）
2) 指出最关键的1-2个可信点
3) 用中文简明解释"""

    return f"""推文内容："{text}"

检测结果：该推文被判定为【{label_text}】，置信度 {confidence:.2%}。

{hint}

请给出判断依据（中文2-3句话）："""


def build_user_prompt_fewshot(text: str, label: int, confidence: float) -> str:
    """Few-shot 版：包含示例"""
    label_text = "谣言" if label == 1 else "非谣言"

    # 选择同标签的示例（最多2个）
    examples = [e for e in FEW_SHOT_EXAMPLES if e["label"] == label][:2]

    parts = ["以下是一些推文分析的示例：\n"]
    for i, ex in enumerate(examples, 1):
        ex_label = "谣言" if ex["label"] == 1 else "非谣言"
        parts.append(f"示例{i}：推文 \"{ex['text']}\" → {ex_label}")
        parts.append(f"解释：{ex['explanation']}\n")

    parts.append("---")
    parts.append(f'现在请分析推文："{text}"')
    parts.append(f"检测结果：【{label_text}】，置信度 {confidence:.2%}")
    parts.append("请用与示例相同的格式给出解释：")

    return "\n".join(parts)


def build_user_prompt_rag(
    text: str, label: int, confidence: float, retrieved: "list[dict] | None" = None,
) -> str:
    """RAG 增强版：附带检索到的相似案例作为参考"""
    label_text = "谣言" if label == 1 else "非谣言"

    parts = [f'推文内容："{text}"']
    parts.append(f"检测结果：该推文被判定为【{label_text}】，置信度 {confidence:.2%}。")

    if retrieved:
        parts.append("\n以下是从训练集中检索到的相似推文（仅供参考）：")
        for i, r in enumerate(retrieved, 1):
            parts.append(
                f"  {i}. \"{r['text'][:120]}\" "
                f"→ {r['label']}（相似度 {r['similarity']:.2f}）"
            )

    parts.append("\n请结合推文内容和参考案例，用中文给出2-3句判断依据：")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# 兼容旧接口
# ═══════════════════════════════════════════════════════════════

LABEL_TEXT = {0: "非谣言", 1: "谣言"}
LABEL_MAP = LABEL_TEXT

# 默认使用的版本
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT_V2
DEFAULT_BUILD_USER_PROMPT = build_user_prompt_v1


def build_prompt(
    text: str,
    label: int,
    confidence: float,
    prompt_type: str = "v1",
    evidence_tags: "list[str] | None" = None,
) -> str:
    """兼容旧 build_prompt 接口：返回纯文本 user prompt。

    prompt_type: "v0" | "v1" | "fewshot" | "rag"
    """
    builders = {
        "v0": build_user_prompt_v0,
        "v1": build_user_prompt_v1,
        "fewshot": build_user_prompt_fewshot,
        "rag": build_user_prompt_rag,
        # 旧名称兼容
        "basic": build_user_prompt_v0,
        "evidence": build_user_prompt_v1,
        "fewshot_evidence": build_user_prompt_fewshot,
    }
    build_fn = builders.get(prompt_type, DEFAULT_BUILD_USER_PROMPT)
    if prompt_type == "rag":
        return build_fn(text, label, confidence)
    return build_fn(text, label, confidence)


__all__ = [
    "SYSTEM_PROMPT_V1", "SYSTEM_PROMPT_V2",
    "DEFAULT_SYSTEM_PROMPT", "DEFAULT_BUILD_USER_PROMPT",
    "build_user_prompt_v0", "build_user_prompt_v1",
    "build_user_prompt_fewshot", "build_user_prompt_rag",
    "build_prompt", "LABEL_TEXT", "LABEL_MAP",
]
