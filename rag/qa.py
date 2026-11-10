"""问答编排：优先「检索 + 生成」，任何一环不可用时回退到规则匹配，不向外抛异常。"""

from rag import embedder, generate, index, knowledge


def status():
    """返回 (mode, 说明)。mode 为 'rag' 表示检索生成可用。"""
    if not index.index_exists():
        return "rules", "未建索引（先跑 python -m rag.build），问答走规则匹配"
    if not generate.has_key():
        return "rules", "没有 DEEPSEEK_API_KEY，问答走规则匹配"
    if not embedder.is_available():
        return "rules", "没有安装 sentence-transformers，问答走规则匹配"
    meta = index.load_index() or {}
    model = (meta.get("meta") or {}).get("model", embedder.MODEL_NAME)
    return "rag", f"检索 + 生成已就绪（{model} + deepseek-chat）"


def status_text():
    return status()[1]


def _rules(question, default_class, reason=""):
    text = knowledge.rule_answer(question, default_class)
    note = "规则匹配"
    if reason:
        note = f"{note}（{reason}）"
    return {"answer": text, "sources": [], "mode": "rules", "mode_text": note}


def answer(question, default_class=None):
    """返回 {answer, sources, mode, mode_text}。sources 里是命中的知识块。"""
    mode, _ = status()
    if mode != "rag":
        return _rules(question, default_class)

    loaded = index.load_index()
    if not loaded:
        return _rules(question, default_class, "索引读不出来")

    try:
        query_vec = embedder.encode([question], is_query=True)[0]
        hits = index.rank(query_vec, loaded["chunks"], loaded["vectors"])
        if not hits:
            return _rules(question, default_class, "没有检索到相关段落")
        text = generate.generate(question, hits)
        return {
            "answer": text,
            "sources": hits,
            "mode": "rag",
            "mode_text": "检索 + 生成（deepseek-chat）",
        }
    except Exception as e:
        return _rules(question, default_class, f"检索生成失败：{e}")
