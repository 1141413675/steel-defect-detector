import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from collections import Counter

# ---------- 配置 ----------
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(_BASE, "weights", "best.pt"))
CONF_THRESH = 0.25
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# ---------- 模拟缺陷知识库 (RAG 模板) ----------
DEFECT_KNOWLEDGE = {
    "scratches": "**划痕 (scratches)**\n- **成因**: 轧制过程中辊面不干净或异物划伤。\n- **处理**: 检查并清理轧辊，调整工艺参数。",
    "crazing": "**裂纹 (crazing)**\n- **成因**: 表面冷却速度过快或应力集中。\n- **处理**: 控制冷却速率，优化退火工艺。",
    "inclusion": "**夹杂物 (inclusion)**\n- **成因**: 冶炼或浇铸过程中非金属杂质残留。\n- **处理**: 提高钢水纯净度，加强精炼。",
    "patches": "**斑块 (patches)**\n- **成因**: 表面氧化皮脱落不均或局部成分偏析。\n- **处理**: 优化除鳞工艺。",
    "pitted_surface": "**点蚀 (pitted_surface)**\n- **成因**: 局部腐蚀或氧化皮粘附。\n- **处理**: 加强表面酸洗和清理。",
    "rolled-in_scale": "**压入氧化皮 (rolled-in_scale)**\n- **成因**: 加热炉内氧化皮过厚未清除干净被压入钢体。\n- **处理**: 改进高压水除鳞效果。"
}

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
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{names[cls]} {conf:.2f}", (x1, max(y1-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

# ---------- 页面 ----------
st.set_page_config(page_title="钢材表面缺陷检测", layout="wide")
st.title("🔧 钢材表面缺陷检测系统")
st.caption("YOLOv8n · NEU-DET · 6 类缺陷")

model, model_err = load_model(MODEL_PATH)

with st.sidebar:
    st.header("设置")
    conf = st.slider("置信度阈值", 0.05, 0.95, CONF_THRESH, 0.05)
    st.markdown("---")
    st.markdown("**缺陷类别**")
    for c in CLASSES:
        st.markdown(f"- {c}")
    st.markdown("---")
    st.markdown("上传一张钢板表面图片开始检测。")
    if model is None:
        st.markdown("---")
        st.warning(model_err)
        st.caption("把训练好的 best.pt 放进 weights/ 后刷新页面即可检测。")

uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    if model is None:
        st.warning("当前没有可用的模型权重，无法检测。把 best.pt 放进 weights/ 后刷新页面。")
        st.stop()
    img_pil = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    results = model.predict(source=img_bgr, conf=conf, verbose=False)
    res = results[0]
    
    out_bgr = draw_boxes(img_bgr.copy(), res.boxes, model.names)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    
    boxes = res.boxes
    cls_list = [int(b.cls[0]) for b in boxes] if len(boxes) else []
    names_detected = [model.names[c] for c in cls_list]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原图")
        st.image(img_pil, use_container_width=True)
    with col2:
        st.subheader("检测结果")
        st.image(out_rgb, use_container_width=True)
        
    st.markdown("---")
    if len(boxes) == 0:
        st.info("未检测到缺陷（可尝试降低置信度阈值）。")
    else:
        st.success(f"共检测到 **{len(boxes)}** 个缺陷目标")
        cnt = Counter(names_detected)
        cols = st.columns(len(CLASSES))
        for i, c in enumerate(CLASSES):
            with cols[i]:
                st.metric(c, cnt.get(c, 0), border=True)
                
        rows = []
        for b in boxes:
            rows.append({
                "类别": model.names[int(b.cls[0])],
                "置信度": round(float(b.conf[0]), 4),
                "x1": int(b.xyxy[0][0]), "y1": int(b.xyxy[0][1]),
                "x2": int(b.xyxy[0][2]), "y2": int(b.xyxy[0][3]),
            })
        st.subheader("检测明细")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        
        # ---------- RAG 问答入口 ----------
        st.markdown("---")
        st.subheader("💡 缺陷知识问答 (RAG)")
        detected_classes = list(cnt.keys())
        if detected_classes:
            selected_defect = st.selectbox("选择检测到的缺陷查看详情：", detected_classes)
            if selected_defect:
                st.markdown(DEFECT_KNOWLEDGE.get(selected_defect, "暂无该缺陷信息"))
            
            st.markdown("---")
            user_q = st.text_input("或者输入你的问题（如：划痕怎么处理？）：")
            if user_q:
                # 简单模板匹配模拟 RAG 检索
                for key, value in DEFECT_KNOWLEDGE.items():
                    if key in user_q or ("划痕" in user_q and key=="scratches"):
                        st.write("**检索到相关知识：**")
                        st.markdown(value)
                        break
                else:
                    st.write("（知识库中没有找到相关答案，后续可接入大模型）")
        else:
            st.info("检测到缺陷后，可在此处查看缺陷成因与处理建议。")