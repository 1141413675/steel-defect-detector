cat > README.md << 'EOF'
# 钢材表面缺陷智能检测系统

基于 YOLOv8 + Streamlit + RAG 的工业质检 Demo

## 功能

- 🔍 YOLOv8n 检测钢板表面 6 类缺陷（mAP50=0.754）
- 📊 实时可视化：检测框、置信度、统计面板
- 💡 RAG 知识问答：检测后自动给出缺陷成因与处理建议
- 🐳 Docker 容器化部署

## 缺陷类别

| 类别 | 说明 | mAP50 |
|------|------|-------|
| crazing | 裂纹 | 0.441 |
| inclusion | 夹杂物 | 0.834 |
| patches | 斑块 | 0.922 |
| pitted_surface | 点蚀 | 0.773 |
| rolled-in_scale | 压入氧化皮 | 0.619 |
| scratches | 划痕 | 0.935 |

## 快速启动

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## 训练

\`\`\`python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=100, imgsz=640, batch=16)
\`\`\`

## 项目结构

\`\`\`
├── app.py              # Streamlit 主程序
├── requirements.txt    # Python 依赖
├── Dockerfile          # 容器化配置
├── data.yaml           # 数据集配置
└── weights/            # 训练权重（需自行下载 best.pt）
\`\`\`
EOF