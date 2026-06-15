"""
SJTU LLM API 封装
- OpenAI 兼容接口，用于谣言检测解释生成
- 支持环境变量 SJTU_API_KEY 覆盖默认 API Key
- 无需 .env 文件即可使用默认配置
"""
import os
import time
from pathlib import Path

try:
    import requests
    from requests import RequestException
except ImportError:
    requests = None  # type: ignore[assignment]
    RequestException = OSError  # type: ignore[assignment]


class LLMAPIError(Exception):
    """Raised when the LLM API cannot return a usable response."""


# ── API 配置 ──
# 优先从环境变量读取，否则使用默认值（无需 .env 文件）
API_KEY = os.getenv("SJTU_API_KEY", "sk-Iob3w6kF6wGJaroAv9uaCw")
API_BASE = os.getenv("SJTU_LLM_API_URL", "https://models.sjtu.edu.cn/api/v1")
DEFAULT_MODEL = os.getenv("SJTU_LLM_MODEL", "deepseek-chat")

# 同时支持从 .env 文件加载（覆盖默认值）
def _load_dotenv_if_needed() -> None:
    """Load a local .env file without requiring python-dotenv."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        return

_load_dotenv_if_needed()
# 重新读取，.env 中的值优先
API_KEY = os.getenv("SJTU_API_KEY", API_KEY)
API_BASE = os.getenv("SJTU_LLM_API_URL", API_BASE)
DEFAULT_MODEL = os.getenv("SJTU_LLM_MODEL", DEFAULT_MODEL)


def call_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 256,
    timeout: int = 30,
    max_retries: int = 2,
) -> str:
    """
    调用 SJTU LLM API（OpenAI 兼容接口）

    Args:
        prompt: 用户输入内容
        system_prompt: 系统提示词（角色设定）
        model: 模型名称
        temperature: 生成温度（0-2，越低越确定）
        max_tokens: 最大生成长度
        timeout: 请求超时秒数
        max_retries: 失败重试次数

    Returns:
        模型生成的文本
    """
    if requests is None:
        raise LLMAPIError("The requests package is not installed")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{API_BASE}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            last_error = f"请求超时（{timeout}s）"
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP 错误: {e.response.status_code} - {e.response.text[:200]}"
        except requests.exceptions.RequestException as e:
            last_error = f"请求失败: {e}"
        except (KeyError, IndexError, TypeError) as e:
            last_error = f"响应解析失败: {e}"

        if attempt < max_retries:
            wait = (attempt + 1) * 2
            print(f"[LLM] {last_error}，{wait}s 后重试 ({attempt + 1}/{max_retries})")
            time.sleep(wait)

    raise LLMAPIError(f"LLM 调用失败（已重试{max_retries}次）: {last_error}")


def call_sjtu_llm(
    prompt: "str | list[dict[str, str]]",
    temperature: float = 0.3,
    max_tokens: int = 256,
    timeout: int = 30,
    retries: int = 2,
) -> str:
    """兼容旧接口：支持纯文本或 messages 列表。

    若传入 messages 列表，则提取 system 和 user 消息后调用 call_llm。
    """
    if isinstance(prompt, list):
        system_prompt = ""
        user_prompt = ""
        for msg in prompt:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_prompt = msg["content"]
        if not user_prompt:
            user_prompt = str(prompt)
        return call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=retries,
        )
    return call_llm(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=retries,
    )


__all__ = ["LLMAPIError", "call_llm", "call_sjtu_llm"]


if __name__ == "__main__":
    # 快速连通性测试
    print("测试 SJTU LLM API 连通性...")
    try:
        reply = call_llm(
            prompt="请用一句话介绍上海交通大学。",
            system_prompt="你是一个友好的助手，请用中文回答。",
            max_tokens=100,
        )
        print(f"[OK] API 连通成功！\n回复: {reply}")
    except Exception as e:
        print(f"[FAIL] API 连通失败: {e}")
