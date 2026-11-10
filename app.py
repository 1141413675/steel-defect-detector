import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from core import class_ids, filter_rows, summarize, to_csv
from rag import qa
from rag.knowledge import CLASS_LABELS, CLASS_NAMES

# ---------- 配置 ----------
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(_BASE, "weights", "best.pt"))
CONF_THRESH = 0.25


# ---------- 模型加载 ----------
# 权重缺失或读不出来时不让页面崩掉：返回 None + 原因，界面照常打开
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None, f"没有找到权重文件：{path}"
    try:
        return YOLO(path), ""
    except Exception as e:
        return None, f"权重读不出来：{e}"


def draw_boxes(img, boxes, names):
    """在原图副本上画框，返回画好的图。"""
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{names[cls]} {conf:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img


def predict_rows(model, image_bgr, conf, ids, image_name):
    """单张推理，返回明细行。"""
    res = model.predict(source=image_bgr, conf=conf, classes=ids or None, verbose=False)[0]
    rows = []
    for b in res.boxes:
        cls = int(b.cls[0])
        name = model.names[cls]
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
        rows.append({
            "图片": image_name,
            "类别": name,
            "中文名": CLASS_LABELS.get(name, name),
            "置信度": round(float(b.conf[0]), 4),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
    return res, rows


# ---------- 页面 ----------
st.set_page_config(page_title="钢材表面缺陷检测", layout="wide")
st.title("钢材表面缺陷检测")
st.caption("YOLOv8n · NEU-DET · 6 类缺陷")

model, model_err = load_model(MODEL_PATH)

with st.sidebar:
    st.header("设置")
    conf = st.slider("置信度阈值", 0.05, 0.95, CONF_THRESH, 0.05)
    picked = st.multiselect("只看这些类别", CLASS_NAMES, default=CLASS_NAMES,
                            format_func=lambda n: f"{CLASS_LABELS.get(n, n)}（{n}）")
    st.markdown("---")
    if model is None:
        st.warning(model_err)
        st.caption("把训练好的 best.pt 放进 weights/ 后刷新页面即可检测。问答不受影响。")
    else:
        st.caption("权重已载入。")

tab_detect, tab_qa = st.tabs(["检测", "知识问答"])

with tab_detect:
    files = st.file_uploader(
        "上传钢板图片（可多选）",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )

    if not files:
        st.caption("支持一次选多张，结果会汇总成一张表。")
    elif model is None:
        st.warning("当前没有可用的模型权重，无法检测。把 best.pt 放进 weights/ 后刷新页面。")
    elif not picked:
        st.info("左侧没有选中任何类别，先在「只看这些类别」里选至少一类。")
    else:
        ids = class_ids(picked, model.names)
        all_rows = []
        for f in files:
            img_pil = Image.open(f).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            res, rows = predict_rows(model, img_bgr, conf, ids, f.name)
            marked = draw_boxes(img_bgr.copy(), res.boxes, model.names)
            marked_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
            with st.expander(f"{f.name} · {len(res.boxes)} 处", expanded=len(files) == 1):
                col1, col2 = st.columns(2)
                col1.image(img_pil, caption="原图", use_container_width=True)
                col2.image(marked_rgb, caption="检测结果", use_container_width=True)
            all_rows.extend(rows)

        rows = filter_rows(all_rows, picked)
        if not rows:
            st.info("这批图片没有检出所选类别的缺陷，可以调低置信度阈值再试。")
        else:
            st.subheader("汇总")
            c1, c2, c3 = st.columns(3)
            c1.metric("图片", len(files))
            c2.metric("缺陷", len(rows))
            c3.metric("涉及类别", len(summarize(rows)))
            st.dataframe(pd.DataFrame(summarize(rows)), use_container_width=True, hide_index=True)

            st.subheader("明细")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.download_button(
                "导出 CSV",
                data=to_csv(rows).encode("utf-8-sig"),
                file_name="detections.csv",
                mime="text/csv",
            )

with tab_qa:
    st.caption(qa.status_text())
    target = st.selectbox(
        "缺陷类型",
        CLASS_NAMES,
        format_func=lambda n: f"{CLASS_LABELS.get(n, n)}（{n}）",
    )
    question = st.text_input("问题", placeholder="例如：这种缺陷对冷轧有什么影响，怎么处理？")
    if st.button("提问", type="primary") and question.strip():
        with st.spinner("检索并生成…"):
            ans = qa.answer(question.strip(), default_class=target)
        st.markdown(ans["answer"])
        if ans["sources"]:
            with st.expander(f"引用来源（{len(ans['sources'])} 段）"):
                for i, s in enumerate(ans["sources"], 1):
                    head = s.get("title", "") + (" · " + s.get("heading", "") if s.get("heading") else "")
                    st.markdown(f"**{i}. {head}**（相似度 {s['score']:.3f}）")
                    st.caption(s.get("text", ""))
        st.caption(ans["mode_text"])
