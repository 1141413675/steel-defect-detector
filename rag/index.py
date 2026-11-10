"""本地向量索引：npy 存向量、json 存块与元信息，检索用 numpy 算余弦相似度。

不引入 faiss 之类的重型依赖；六类缺陷的文档只有几十个块，暴力检索足够。
"""

import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "index")
VECTOR_FILE = os.path.join(INDEX_DIR, "vectors.npy")
CHUNK_FILE = os.path.join(INDEX_DIR, "chunks.json")
META_FILE = os.path.join(INDEX_DIR, "meta.json")

TOP_K = 4
MIN_SCORE = 0.30


def normalize(matrix):
    """按行做 L2 归一化，0 向量保持为 0。"""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return arr / norm


def cosine(query_vec, matrix):
    """归一化后的点积就是余弦相似度。"""
    q = normalize(query_vec)
    m = normalize(matrix)
    return (m @ q.T).ravel()


def rank(query_vec, chunks, vectors, top_k=TOP_K, min_score=MIN_SCORE):
    """返回按相似度降序的命中块，附 score。"""
    if len(chunks) == 0 or len(vectors) == 0:
        return []
    scores = cosine(query_vec, vectors)
    order = np.argsort(-scores)[:top_k]
    hits = []
    for i in order:
        score = float(scores[i])
        if score < min_score:
            continue
        item = dict(chunks[int(i)])
        item["score"] = score
        hits.append(item)
    return hits


def save_index(chunks, vectors, meta):
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(VECTOR_FILE, normalize(vectors))
    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def index_exists():
    return os.path.exists(VECTOR_FILE) and os.path.exists(CHUNK_FILE)


def load_index():
    """读索引，不存在或损坏时返回 None，由调用方降级。"""
    if not index_exists():
        return None
    try:
        vectors = np.load(VECTOR_FILE)
        with open(CHUNK_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        meta = {}
        if os.path.exists(META_FILE):
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
        return {"vectors": vectors, "chunks": chunks, "meta": meta}
    except Exception:
        return None
