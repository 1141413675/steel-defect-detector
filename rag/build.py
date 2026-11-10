"""建索引：python -m rag.build

读 knowledge/*.md，切块，用中文 embedding 编码，写到 index/ 下。
需要先装依赖：pip install -r requirements-rag.txt
"""

import glob
import os
import time

from rag import chunking, embedder, index
from rag.knowledge import CLASS_LABELS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")


def iter_documents():
    """按文件名顺序返回 [(title, text)]。"""
    for path in sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        title = CLASS_LABELS.get(stem, stem)
        with open(path, "r", encoding="utf-8") as f:
            yield title, f.read()


def build():
    chunks = []
    for title, text in iter_documents():
        chunks.extend(chunking.chunks_of_document(title, text))
    if not chunks:
        raise SystemExit(f"knowledge/ 里没有可用的文档：{KNOWLEDGE_DIR}")

    vectors = embedder.encode([chunking.embed_text(c) for c in chunks])
    meta = {
        "model": embedder.MODEL_NAME,
        "chunks": len(chunks),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    index.save_index(chunks, vectors, meta)
    print(f"已建索引：{len(chunks)} 块，模型 {embedder.MODEL_NAME}")
    print(f"输出目录：{index.INDEX_DIR}")


if __name__ == "__main__":
    build()
