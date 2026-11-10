"""中文 embedding 封装。

默认用 BAAI/bge-small-zh-v1.5（约 100MB），可用环境变量 EMBED_MODEL 换成
text2vec-base-chinese 等 Sentence-Transformers 兼容模型。
sentence-transformers 只在真正编码时导入，没装依赖时抛 RuntimeError，由上层降级。
"""

import os
from functools import lru_cache

MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

# bge 系列建议给查询加指令前缀，文档侧不加
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


@lru_cache(maxsize=1)
def _model():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # 依赖没装
        raise RuntimeError(f"没有安装 sentence-transformers：{e}")
    return SentenceTransformer(MODEL_NAME)


def is_available():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def encode(texts, is_query=False):
    """把文本编码成向量（list[list[float]]）。"""
    model = _model()
    if isinstance(texts, str):
        texts = [texts]
    payload = [QUERY_PREFIX + t for t in texts] if is_query else list(texts)
    vectors = model.encode(payload, normalize_embeddings=True, show_progress_bar=False)
    return [list(map(float, v)) for v in vectors]
