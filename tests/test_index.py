"""向量检索的数学部分测试：归一化、余弦、排序与阈值。"""

import numpy as np

from rag import index

CHUNKS = [
    {"title": "裂纹", "heading": "成因", "text": "冷却不均"},
    {"title": "划痕", "heading": "成因", "text": "辊面异物"},
    {"title": "点蚀", "heading": "成因", "text": "残酸腐蚀"},
]
VECTORS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


def test_normalize_unit_length():
    out = index.normalize([[3.0, 4.0]])
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_normalize_zero_vector_safe():
    out = index.normalize([[0.0, 0.0]])
    assert np.allclose(out, [[0.0, 0.0]])


def test_cosine_orthogonal_and_parallel():
    scores = index.cosine([1.0, 0.0, 0.0], VECTORS)
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.0)


def test_rank_orders_by_similarity():
    hits = index.rank([0.9, 0.1, 0.0], CHUNKS, VECTORS, min_score=0.0)
    assert hits[0]["title"] == "裂纹"
    assert hits[0]["score"] > hits[1]["score"]


def test_rank_respects_top_k():
    hits = index.rank([1.0, 1.0, 1.0], CHUNKS, VECTORS, top_k=2, min_score=0.0)
    assert len(hits) == 2


def test_rank_filters_low_scores():
    hits = index.rank([1.0, 0.0, 0.0], CHUNKS, VECTORS, min_score=0.5)
    assert [h["title"] for h in hits] == ["裂纹"]


def test_rank_empty_index():
    assert index.rank([1.0, 0.0, 0.0], [], [], min_score=0.0) == []


def test_rank_does_not_mutate_chunks():
    before = [dict(c) for c in CHUNKS]
    index.rank([1.0, 0.0, 0.0], CHUNKS, VECTORS, min_score=0.0)
    assert CHUNKS == before
