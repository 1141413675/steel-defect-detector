"""生成侧的纯函数测试：key 读取与资料拼接（不调用接口）。"""

from rag import generate


def test_has_key_false_without_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert generate.has_key() is False


def test_has_key_ignores_blank(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "   ")
    assert generate.has_key() is False


def test_has_key_true(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert generate.has_key() is True


def test_build_context_numbering_and_headings():
    hits = [
        {"title": "裂纹", "heading": "常见成因", "text": "冷却不均"},
        {"title": "划痕", "heading": "", "text": "辊面异物"},
    ]
    text = generate.build_context(hits)
    assert "[1] 裂纹 · 常见成因" in text
    assert "[2] 划痕" in text
    assert "冷却不均" in text and "辊面异物" in text


def test_build_context_empty():
    assert generate.build_context([]) == ""
