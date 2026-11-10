"""六类缺陷的标签、关键词与规则兜底文本。

规则兜底用于没有 API key、没有索引、或者检索链路出错时的降级回答，
内容刻意保持短小，真正的知识在 knowledge/*.md 里。
"""

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_LABELS = {
    "crazing": "裂纹",
    "inclusion": "夹杂物",
    "patches": "斑块",
    "pitted_surface": "点蚀",
    "rolled-in_scale": "压入氧化皮",
    "scratches": "划痕",
}

FALLBACK_KNOWLEDGE = {
    "scratches": "**划痕 (scratches)**\n- **成因**: 轧制过程中辊面不干净或异物划伤。\n- **处理**: 检查并清理轧辊，调整工艺参数。",
    "crazing": "**裂纹 (crazing)**\n- **成因**: 表面冷却速度过快或应力集中。\n- **处理**: 控制冷却速率，优化退火工艺。",
    "inclusion": "**夹杂物 (inclusion)**\n- **成因**: 冶炼或浇铸过程中非金属杂质残留。\n- **处理**: 提高钢水纯净度，加强精炼。",
    "patches": "**斑块 (patches)**\n- **成因**: 表面氧化皮脱落不均或局部成分偏析。\n- **处理**: 优化除鳞工艺。",
    "pitted_surface": "**点蚀 (pitted_surface)**\n- **成因**: 局部腐蚀或氧化皮粘附。\n- **处理**: 加强表面酸洗和清理。",
    "rolled-in_scale": "**压入氧化皮 (rolled-in_scale)**\n- **成因**: 加热炉内氧化皮过厚未清除干净被压入钢体。\n- **处理**: 改进高压水除鳞效果。",
}

# 提问里可能出现的说法，用于规则兜底时判断问的是哪一类
KEYWORDS = {
    "crazing": ["裂纹", "龟裂", "网状裂", "crazing"],
    "inclusion": ["夹杂", "杂质", "非金属", "inclusion"],
    "patches": ["斑块", "色差", "patches"],
    "pitted_surface": ["点蚀", "麻点", "凹坑", "pitted"],
    "rolled-in_scale": ["氧化皮", "压入", "scale"],
    "scratches": ["划痕", "刮伤", "划伤", "scratches"],
}


def match_class(text):
    """在问题里找缺陷类型关键词，找不到返回 None。"""
    if not text:
        return None
    lowered = text.lower()
    for name, words in KEYWORDS.items():
        for w in words:
            if w in lowered:
                return name
    return None


def rule_answer(question, default_class=None):
    """规则兜底回答：优先按问题里的关键词匹配，其次用界面上选中的类别。"""
    name = match_class(question) or default_class
    if name in FALLBACK_KNOWLEDGE:
        return FALLBACK_KNOWLEDGE[name]
    return "知识库里只有 6 类缺陷（" + "、".join(CLASS_LABELS[n] for n in CLASS_NAMES) + "），换个问法或直接选类别。"
