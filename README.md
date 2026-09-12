# 钢材表面缺陷智能检测系统

> 基于 **YOLOv8 + Streamlit + RAG** 的工业质检 Demo，支持 6 类钢板表面缺陷的实时检测与智能问答。

![Image](%E6%88%AA%E5%B1%8F2026-09-12%2011.58.21.png)

---

## ✨ 功能亮点

- 🔍 **YOLOv8n 目标检测**：单图推理 < 0.1s（M1 Mac），mAP50 = **0.754**
- 🖼️ **可视化结果**：检测框 + 类别 + 置信度，原图 / 结果图并排对比
- 📊 **数据统计**：缺陷计数卡片 + 检测明细表格
- 💡 **RAG 知识问答**：选择缺陷类型，自动给出**成因分析与处理建议**
- 🐳 **Docker 一键部署**：`docker run` 即可启动完整服务

## 🎯 支持的缺陷类别（NEU-DET 数据集）

| 类别 | 中文名 | mAP50 |
|------|--------|-------|
| crazing | 裂纹 | 0.441 |
| inclusion | 夹杂物 | 0.834 |
| patches | 斑块 | 0.922 |
| pitted_surface | 点蚀 | 0.773 |
| rolled-in_scale | 压入氧化皮 | 0.619 |
| scratches | 划痕 | 0.935 |

## 🖥️ 界面预览

### 初始界面
左侧为设置栏（置信度阈值滑块 + 6 类缺陷列表），右侧为图片上传区。

![初始界面](%E6%88%AA%E5%B1%8F2026-09-12%2011.57.17.png)

### 检测效果
上传钢板图片后，模型自动框出缺陷并标注类别与置信度。

![检测效果](%E6%88%AA%E5%B1%8F2026-09-12%2011.58.21.png)

### 数据统计 + RAG 问答
底部展示缺陷计数统计、检测明细表格，以及基于 RAG 的缺陷成因与处理建议。

![RAG 知识问答](%E6%88%AA%E5%B1%8F2026-09-12%2011.58.32.png)

## 🚀 快速启动

### 方式一：本地运行

1. 克隆仓库

git clone https://gitee.com/chen-liangana/steel-defect-detector.git

cd steel-defect-detector

2. 创建虚拟环境

python3 -m venv venv

source venv/bin/activate

3. 安装依赖

pip install -r requirements.txt

4. 放入模型权重

将训练好的 best.pt 放到 weights/ 目录下

mkdir -p weights

cp /path/to/best.pt weights/

5. 启动服务

streamlit run app.py

浏览器打开 `http://localhost:8501` 即可使用。

### 方式二：Docker 部署

docker build -t steel-defect-detector .

docker run -p 8501:8501 steel-defect-detector

## 🏋️ 模型训练

python

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(

data="data.yaml",

epochs=100,

imgsz=640,

batch=16,

name="steel_yolov8n"

)

训练完成后将最佳权重 `runs/detect/steel_yolov8n/weights/best.pt` 放入 `weights/` 目录。

## 📁 项目结构

steel-defect-detector/

├── app.py              # Streamlit 主程序

├── data.yaml           # 数据集配置

├── requirements.txt    # Python 依赖

├── Dockerfile          # 容器化配置

├── .gitignore          # Git 忽略规则

├── docs/               # 文档与截图

└── weights/            # 模型权重（不入库，需自行放置 best.pt）

## 🛠️ 技术栈

| 模块 | 技术 |
|------|------|
| 目标检测 | YOLOv8 (Ultralytics) |
| 前端界面 | Streamlit |
| 知识问答 | RAG (Retrieval-Augmented Generation) |
| 图像处理 | OpenCV / Pillow |
| 部署 | Docker |

## 📄 License

MIT License
EOF