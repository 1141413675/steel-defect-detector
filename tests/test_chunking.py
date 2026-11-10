"""知识文档切块的纯函数测试。"""

from rag import chunking

SAMPLE = """# 裂纹（crazing）

开头的一段说明。

## 外观特征

- 呈细密网状
- 单条纹路很浅

## 常见成因

第一段成因。

第二段成因，比较长，用来验证段落不会被切断。
"""


def test_split_sections_uses_h2():
    sections = chunking.split_sections(SAMPLE)
    headings = [h for h, _ in sections]
    assert headings == ["", "外观特征", "常见成因"]


def test_split_sections_drops_h1_and_empty():
    sections = dict(chunking.split_sections(SAMPLE))
    assert "裂纹（crazing）" not in sections
    assert sections["外观特征"].startswith("- 呈细密网状")


def test_pack_keeps_paragraphs_whole():
    paragraphs = ["a" * 250, "b" * 250, "c" * 250]
    chunks = chunking.pack(paragraphs, max_chars=420)
    assert len(chunks) == 3
    assert sorted(chunks) == sorted(paragraphs)


def test_pack_merges_short_paragraphs():
    chunks = chunking.pack(["短句一", "短句二"], max_chars=100)
    assert len(chunks) == 1
    assert "短句一" in chunks[0] and "短句二" in chunks[0]


def test_pack_empty():
    assert chunking.pack([]) == []


def test_chunks_of_document_fields():
    chunks = chunking.chunks_of_document("裂纹", SAMPLE)
    assert chunks
    for c in chunks:
        assert c["title"] == "裂纹"
        assert c["heading"] in ("", "外观特征", "常见成因")
        assert c["text"]


def test_embed_text_contains_context():
    item = {"title": "裂纹", "heading": "常见成因", "text": "正文"}
    text = chunking.embed_text(item)
    assert text.startswith("裂纹 · 常见成因")
    assert text.endswith("正文")


def test_embed_text_without_heading():
    item = {"title": "裂纹", "heading": "", "text": "正文"}
    assert chunking.embed_text(item) == "裂纹\n正文"
