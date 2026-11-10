"""调 DeepSeek 生成回答（OpenAI 兼容接口）。

key 只从环境变量 DEEPSEEK_API_KEY 读，不写进仓库、不打印。
"""

import os

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

SYSTEM_PROMPT = (
    "你是钢铁厂表面质量工程师。只依据给出的资料回答，不要引入资料以外的具体数字或结论。"
    "回答分三部分：可能原因、对下游工序的影响、建议的处置措施，每条尽量短。"
    "引用了哪段资料就在句末标注编号，例如 [1]。资料不足以回答时直接说资料不足，不要猜。"
)


def has_key():
    return bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())


def build_context(hits):
    lines = []
    for i, h in enumerate(hits, 1):
        head = h.get("title", "") + (" · " + h.get("heading", "") if h.get("heading") else "")
        lines.append(f"[{i}] {head}\n{h.get('text', '')}")
    return "\n\n".join(lines)


def generate(question, hits, timeout=60):
    """返回模型回答文本；接口异常时抛 RuntimeError，由上层降级。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("没有配置 DEEPSEEK_API_KEY")
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"没有安装 openai 客户端：{e}")

    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=timeout)
    user_prompt = f"资料：\n{build_context(hits)}\n\n问题：{question}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("模型返回了空回答")
    return text
