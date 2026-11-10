"""检测结果的后处理：类别下标、筛选、汇总、导出 CSV。

只放不依赖 Streamlit / OpenCV / 模型的纯函数，方便单元测试。
"""

import csv
import io

HEADERS = ["图片", "类别", "中文名", "置信度", "x1", "y1", "x2", "y2"]


def class_ids(selected, names):
    """把选中的类别名换成模型里的类别下标。

    names 可以是 dict（id -> 类名）或 list（下标即 id）。未选中任何类别时返回空列表，
    调用方据此决定是否把 classes 传给模型。
    """
    wanted = set(selected or [])
    if isinstance(names, dict):
        pairs = names.items()
    else:
        pairs = enumerate(names)
    return [int(i) for i, n in pairs if n in wanted]


def filter_rows(rows, classes):
    """只保留指定类别的行；classes 为空时原样返回。"""
    if not classes:
        return list(rows)
    keep = set(classes)
    return [r for r in rows if r.get("类别") in keep]


def summarize(rows):
    """按类别汇总：数量、平均置信度、涉及图片数，按数量降序。"""
    stat = {}
    for r in rows:
        name = r.get("类别")
        item = stat.get(name)
        if item is None:
            item = {
                "类别": name,
                "中文名": r.get("中文名", name),
                "数量": 0,
                "置信度和": 0.0,
                "图片": set(),
            }
            stat[name] = item
        item["数量"] += 1
        item["置信度和"] += float(r.get("置信度", 0.0))
        item["图片"].add(r.get("图片", ""))

    out = []
    for item in stat.values():
        count = item["数量"]
        out.append({
            "类别": item["类别"],
            "中文名": item["中文名"],
            "数量": count,
            "平均置信度": round(item["置信度和"] / count, 3) if count else 0.0,
            "涉及图片": len(item["图片"]),
        })
    out.sort(key=lambda x: (-x["数量"], x["类别"]))
    return out


def total_count(rows):
    return len(rows)


def to_csv(rows):
    """把明细行导成 CSV 文本，列固定，多余的键忽略。"""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=HEADERS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({h: r.get(h, "") for h in HEADERS})
    return buf.getvalue()
