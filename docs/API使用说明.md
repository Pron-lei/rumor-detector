# SJTU LLM API 使用说明

> 参考文档：https://claw.sjtu.edu.cn/guide/sjtu-api/

## 1. 获取 API Key

1. 访问 [claw.sjtu.edu.cn](https://claw.sjtu.edu.cn) 登录
2. 进入个人设置 → API Keys
3. 创建新的 API Key 并妥善保存

## 2. API 基础信息

- **接口地址**：`https://claw.sjtu.edu.cn/v1/chat/completions`（以官方文档为准）
- **认证方式**：Bearer Token（API Key）
- **请求格式**：OpenAI 兼容格式

## 3. Python 调用示例

```python
import requests

API_KEY = "your-api-key-here"
API_BASE = "https://claw.sjtu.edu.cn/v1"

def call_llm(prompt: str, system_prompt: str = "", model: str = "gpt-4o") -> str:
    """调用 SJTU LLM API 生成解释"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = requests.post(
        f"{API_BASE}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.3,   # 降低随机性，使输出更一致
            "max_tokens": 256      # 解释不宜过长
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# === 谣言检测解释专用调用 ===

SYSTEM_PROMPT = """你是一个专业的谣言检测分析助手。你的任务是根据给定的推文内容和检测结果，用中文解释判断依据。

要求：
1. 引用推文中的具体词语或句式作为证据
2. 解释要有逻辑性，清晰地说明为什么判定为谣言或非谣言
3. 解释限制在2-3句话内，直接、客观
4. 不要编造推文中不存在的信息"""

def generate_explanation(text: str, label: int, confidence: float) -> str:
    """根据推文和检测结果生成判断依据"""
    label_text = "谣言" if label == 1 else "非谣言"
    
    user_prompt = f"""请分析以下推文：

推文内容："{text}"

检测结果：该推文被判定为【{label_text}】，置信度 {confidence:.2%}

请解释为什么这条推文是{label_text}："""
    
    return call_llm(user_prompt, SYSTEM_PROMPT)


# 测试
if __name__ == "__main__":
    text = "BREAKING: Massive earthquake hits downtown, thousands feared dead!"
    explanation = generate_explanation(text, label=1, confidence=0.92)
    print(f"推文: {text}")
    print(f"解释: {explanation}")
```

## 4. 可用模型

| 模型 | 适用场景 |
|------|----------|
| gpt-4o | 推荐主力，推理能力强 |
| gpt-4o-mini | 速度快、成本低，适合批量 |
| 其他模型 | 以平台实际可用列表为准 |

## 5. 注意事项

- API Key 不要提交到 Git 仓库，使用环境变量管理
- 可通过 `os.getenv("SJTU_API_KEY")` 读取
- 注意 API 调用频率限制，批量推理时添加适当间隔
- `temperature` 建议设 0.1-0.3，保证解释的一致性和稳定性
- 单条推理超时设 30 秒，批量时可适当延长
