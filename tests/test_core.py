"""core.py 的纯函数测试。"""

import csv
import io

from core import HEADERS, class_ids, filter_rows, summarize, to_csv, total_count

NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}


def _row(image, cls, conf, label="x"):
    return {"图片": image, "类别": cls, "中文名": label, "置信度": conf,
            "x1": 0, "y1": 0, "x2": 10, "y2": 10}


def test_class_ids_keeps_model_order():
    assert class_ids(["scratches", "crazing"], NAMES) == [0, 5]


def test_class_ids_empty_selection_returns_empty():
    assert class_ids([], NAMES) == []
    assert class_ids(None, NAMES) == []


def test_class_ids_accepts_list_names():
    assert class_ids(["b"], ["a", "b", "c"]) == [1]


def test_filter_rows_by_class():
    rows = [_row("a.jpg", "crazing", 0.9), _row("a.jpg", "patches", 0.8)]
    kept = filter_rows(rows, ["patches"])
    assert [r["类别"] for r in kept] == ["patches"]


def test_filter_rows_empty_keeps_all():
    rows = [_row("a.jpg", "crazing", 0.9)]
    assert filter_rows(rows, []) == rows


def test_summarize_counts_and_average():
    rows = [
        _row("a.jpg", "scratches", 0.8, "划痕"),
        _row("a.jpg", "scratches", 0.6, "划痕"),
        _row("b.jpg", "crazing", 0.5, "裂纹"),
    ]
    out = summarize(rows)
    assert out[0]["类别"] == "scratches"
    assert out[0]["数量"] == 2
    assert out[0]["平均置信度"] == 0.7
    assert out[0]["涉及图片"] == 1
    assert out[1]["涉及图片"] == 1


def test_summarize_empty():
    assert summarize([]) == []


def test_total_count():
    assert total_count([_row("a", "crazing", 0.5)] * 3) == 3


def test_to_csv_header_and_rows():
    rows = [_row("a.jpg", "crazing", 0.9123, "裂纹")]
    rows[0]["多余字段"] = "应被忽略"
    text = to_csv(rows)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert text.splitlines()[0] == ",".join(HEADERS)
    assert len(parsed) == 1
    assert parsed[0]["类别"] == "crazing"
    assert parsed[0]["中文名"] == "裂纹"
    assert "多余字段" not in parsed[0]


def test_to_csv_quotes_chinese_comma():
    rows = [_row("图,1.jpg", "crazing", 0.5, "裂纹")]
    text = to_csv(rows)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert parsed[0]["图片"] == "图,1.jpg"
