"""把 knowledge/*.md 切成语义块。

纯函数，不依赖 embedding 与向量库，方便单元测试。
切分规则：先按二级标题（## ）分节，再把每节正文按空行分段、
合并到不超过 max_chars 的块，块头带上文档标题与节标题作为上下文。
"""

MAX_CHARS = 420


def split_sections(text):
    """按二级标题切节，返回 [(heading, body)]，首个标题之前的内容 heading 为空串。"""
    sections = []
    heading = ""
    buf = []
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append((heading, "\n".join(buf).strip()))
            heading = line[3:].strip()
            buf = []
        elif line.startswith("# "):
            # 文档大标题不进正文
            continue
        else:
            buf.append(line)
    sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def _paragraphs(body):
    blocks = []
    cur = []
    for line in body.splitlines():
        if line.strip() == "":
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


def pack(paragraphs, max_chars=MAX_CHARS):
    """把段落顺序合并成接近 max_chars 的块，不在段落中间截断。"""
    chunks = []
    cur = ""
    for p in paragraphs:
        if not cur:
            cur = p
        elif len(cur) + len(p) + 2 <= max_chars:
            cur = cur + "\n\n" + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def chunks_of_document(title, text, max_chars=MAX_CHARS):
    """整篇文档切成 [{title, heading, text}]。"""
    out = []
    for heading, body in split_sections(text):
        for piece in pack(_paragraphs(body), max_chars=max_chars):
            out.append({
                "title": title,
                "heading": heading,
                "text": piece,
            })
    return out


def embed_text(item):
    """送去编码的文本：拼上标题与节标题，避免块脱离上下文。"""
    head = item["title"] + (" · " + item["heading"] if item["heading"] else "")
    return head + "\n" + item["text"]
