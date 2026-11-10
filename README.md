# 钢材表面缺陷智能检测系统

> YOLOv8n + Streamlit 的钢板表面缺陷检测 Demo：批量上传钢板图像，框出 6 类缺陷并汇总成一张结果表；缺陷问答走「中文向量检索 + DeepSeek 生成」，并在回答里给出引用段落。

![检测效果](docs/detect.png)

## 功能

- **YOLOv8n 目标检测**：识别 NEU-DET 的 6 类表面缺陷，单图推理（M1 Mac）小于 0.1s
- **多图批量检测**：一次上传多张，逐张给出原图与检测结果，最后汇总成一张统计表
- **阈值与类别筛选**：侧栏调置信度阈值，并可按类别只看关心的缺陷（筛选直接传给模型，不是只过滤展示）
- **导出 CSV**：明细（图片、类别、中文名、置信度、框坐标）一键下载，UTF-8 BOM 可直接用 Excel 打开
- **知识问答（检索 + 生成）**：知识文档切块 → bge-small-zh 编码 → 本地向量索引 → 余弦检索 Top-4 → DeepSeek 生成，回答带引用来源
- **降级不崩**：没有 API key、没建索引、依赖缺失或接口报错时，自动回退到规则匹配；没有权重时检测不可用但页面与问答照常

## 缺陷类别与指标

| 类别 | 中文名 | mAP50 |
|------|--------|-------|
| crazing | 裂纹 | 0.441 |
| inclusion | 夹杂物 | 0.834 |
| patches | 斑块 | 0.922 |
| pitted_surface | 点蚀 | 0.773 |
| rolled-in_scale | 压入氧化皮 | 0.619 |
| scratches | 划痕 | 0.935 |

总体 mAP50 = 0.754。

以上数字取自训练结束时的验证集评估输出。该次训练的 `runs/` 目录（`results.csv`、`results.png`、`confusion_matrix.png` 等）**没有随本仓库提交**，所以仓库内没有可直接查看的原始产物。要核对数字，请按下文「评估」一节在相同的数据划分下重跑 `eval.py`；结果会因数据划分、随机种子和环境差异略有出入。

## 界面预览

### 初始界面

左侧为设置栏（置信度阈值、类别筛选），右侧为图片上传区。

![初始界面](docs/ui-home.png)

### 检测效果

上传钢板图片后，模型自动框出缺陷并标注类别与置信度，同时给出原图与结果图的并排对比。

![检测效果](docs/detect.png)

### 数据统计与知识问答

底部展示缺陷计数与检测明细表，问答页给出带引用的回答。

![数据统计与知识问答](docs/stats-knowledge.png)

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/1141413675/steel-defect-detector.git
cd steel-defect-detector
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

三份依赖按需安装：

| 文件 | 用途 | 内容 |
|------|------|------|
| `requirements.txt` | 检测 Demo 与页面 | streamlit、ultralytics、opencv-python、numpy、pillow、pandas |
| `requirements-rag.txt` | 检索问答 | sentence-transformers、openai |
| `requirements-dev.txt` | 测试与 CI | pytest、numpy |

```bash
pip install -r requirements.txt
pip install -r requirements-rag.txt   # 只想跑检测可以跳过
```

依赖没有锁定版本（训练与演示时所用的具体版本未记录），安装时由 pip 解析当前可用版本。遇到冲突时优先固定 `ultralytics`、`streamlit` 与 `sentence-transformers`。

### 4. 准备权重

本仓库不包含模型权重（`weights/` 在 `.gitignore` 中）。需要通过下面的「模型训练」自行训练，或向作者索取 `best.pt`，放到 `weights/best.pt`；也可以用环境变量指定：

```bash
export MODEL_PATH=/path/to/best.pt
```

权重缺失时页面仍能正常打开，侧栏与检测页会提示「没有找到权重文件」，问答不受影响。

### 5. 建知识索引（问答需要）

```bash
python -m rag.build
```

首次运行会下载 `BAAI/bge-small-zh-v1.5`（约 100MB）。产物写到 `index/`（已被 `.gitignore` 忽略），包括 `vectors.npy`、`chunks.json`、`meta.json`。换 embedding 模型时用环境变量 `EMBED_MODEL` 指定，并重新建索引。

### 6. 配置生成模型的 key

```bash
export DEEPSEEK_API_KEY=sk-...
```

key 只从环境变量读取，仓库里没有任何硬编码；没有配置时问答自动回退到规则匹配，不会报错。

### 7. 启动

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

## 知识问答是怎么走的

1. **切块**：`knowledge/*.md` 每类一篇，含外观特征、常见成因、对下游工序的影响、处置建议四节。按二级标题分节，再把段落合并成不超过 420 字的块，块头拼上「类别 · 小节」作为上下文（`rag/chunking.py`）。
2. **编码**：`rag/embedder.py` 用 Sentence-Transformers 加载中文模型（默认 `BAAI/bge-small-zh-v1.5`，可换 `text2vec-base-chinese`），查询侧加 bge 建议的指令前缀。
3. **索引**：向量归一化后存 `index/vectors.npy`，块存 `index/chunks.json`，不引入 faiss——几十个块用 numpy 暴力检索就够（`rag/index.py`）。
4. **检索**：余弦相似度取 Top-4，低于 0.30 的丢弃；没有命中就回退规则。
5. **生成**：`rag/generate.py` 走 OpenAI 兼容接口（`https://api.deepseek.com`，`deepseek-chat`），提示词要求只依据资料作答、按编号引用、资料不足直接说不足。
6. **编排与降级**：`rag/qa.py` 依次检查索引、key、依赖；任何一步失败都会给出可读原因并回退到原来的规则匹配，页面不崩。

回答下方的「引用来源」展开区会列出命中的文档段落、小节标题与相似度，便于核对模型有没有编。

### 实测样例

下面三次问答是本地实跑的结果（索引 30 块，`bge-small-zh-v1.5` 检索 + `deepseek-chat` 生成），编号对应「引用来源」里的段落。

**问：压入氧化皮怎么预防？**

命中：[1] 压入氧化皮·常见成因（0.700）、[2] 压入氧化皮·处置建议（0.695）、[3] 外观特征（0.676）、[4] 首段（0.628）

答（节选）：

> **建议的处置措施（预防）**
> - 提高高压水除鳞压力并逐道检查喷嘴工作情况 [2]
> - 控制加热炉炉内残氧与出钢温度，缩短在炉时间 [2]
> - 优化粗轧至精轧的传搁时间，减少二次氧化皮生成 [2]
>
> **对下游工序的影响**
> - 资料不足（仅说明属热轧工序典型问题 [4]，未给出对下游工序的影响）

**问：辊面异物划伤钢板怎么处理？**

命中：[1] 划痕·常见成因（0.718）、[2] 划痕·处置建议（0.707）、[3] 划痕·首段（0.682）、[4] 压入氧化皮·对下游工序的影响（0.578）

答（节选）：

> - 停机检查辊面与导板，清除粘附物，必要时更换衬板 [2]。
> - 浅划痕可修磨或酸洗改善，深划痕按判废处理；同时按位置反查具体设备段 [2]。
>
> **对下游工序的影响**
> - 资料不足（[4] 讲的是压入氧化皮对下游的影响，未涉及划痕对下游工序的影响）。

**问：裂纹对冷轧有什么影响？**

命中：[1] 裂纹·对下游工序的影响（0.709）、[2] 裂纹·常见成因（0.705）、[3] 裂纹·首段（0.664）、[4] 压入氧化皮·对下游工序的影响（0.653）

答（节选）：

> - 冷轧时裂纹成为断带的起裂点，薄规格尤其明显 [1]。
> - 酸洗时裂纹内部残留酸液，形成后续锈点与点蚀 [1]。
> - 涂镀层在裂纹处附着力下降，后续折弯或冲压时起皮、开裂 [1]。

三次都命中正确类别；问到资料里没写的内容（裂纹的处置措施、划痕对下游的影响）时，模型直接说「资料不足」，没有编。

## Docker 部署

```bash
docker build -t steel-defect-detector .
docker run -p 8501:8501 steel-defect-detector
```

镜像内不包含权重与索引，运行时挂载进去：

```bash
docker run -p 8501:8501 \
  -v "$PWD/weights:/app/weights" \
  -v "$PWD/index:/app/index" \
  -e DEEPSEEK_API_KEY \
  steel-defect-detector
```

## 模型训练

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="steel_yolov8n",
)
```

训练完成后，把 `runs/detect/steel_yolov8n/weights/best.pt` 放到 `weights/` 目录。仓库里没有训练脚本的封装，上面这段就是训练时使用的入口，数据集路径由 `data.yaml` 指定。

## 评估

```bash
python eval.py --weights weights/best.pt --data data.yaml
```

脚本会调用 Ultralytics 的 `val()`，打印总体 mAP50、mAP50-95 和每个类别的 mAP50，与上表的统计口径一致。

验证集的划分在 `data.yaml` 中通过 `images/train` 与 `images/val` 两个目录区分；本仓库没有记录训练时使用的具体划分比例与随机种子，复现时请沿用你自己的划分，并在对比数字时注意这一点。

## 测试与 CI

```bash
pip install -r requirements-dev.txt
pytest -q
```

测试只覆盖纯函数：结果汇总、类别筛选、CSV 导出（`tests/test_core.py`），文档切块（`tests/test_chunking.py`），归一化、余弦与检索排序（`tests/test_index.py`）。不依赖模型权重、网络或 API key。

GitHub Actions（`.github/workflows/ci.yml`）在每次 push 与 PR 时执行：安装 `requirements-dev.txt` → `py_compile` 全部 Python 文件 → `pytest`。运行期的重依赖（torch、ultralytics、sentence-transformers）不装，因为编译与纯函数测试不需要它们。

## 数据

- 数据集：NEU-DET（Northeastern University 钢材表面缺陷数据集），6 类、常用于缺陷检测研究
- 本仓库不附带数据。获取后按下面的结构放到项目根目录的 `datasets/neu_det/`（`data.yaml` 使用相对路径指向这里）：

```
datasets/neu_det/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

- 数据集的获取方式与使用许可以数据集发布方的说明为准（常见途径是在 Kaggle 搜索 NEU-DET，或查阅该数据集原始论文的配套资源）。本仓库的 MIT 许可仅覆盖代码与知识文档，不覆盖数据。

## 项目结构

```
steel-defect-detector/
├── app.py                    # Streamlit 主程序：多图检测、筛选、导出、问答
├── core.py                   # 纯函数：类别映射、筛选、汇总、CSV
├── eval.py                   # 在验证集上重跑指标
├── data.yaml                 # NEU-DET 数据集配置（相对路径）
├── requirements.txt          # 检测与页面依赖
├── requirements-rag.txt      # 检索问答依赖
├── requirements-dev.txt      # 测试依赖
├── Dockerfile                # 容器化配置
├── LICENSE                   # MIT
├── knowledge/                # 六类缺陷知识文档（检索语料）
├── rag/
│   ├── knowledge.py          # 类别标签、关键词、规则兜底文本
│   ├── chunking.py           # 文档切块（纯函数）
│   ├── embedder.py           # 中文 embedding 封装
│   ├── index.py              # 本地向量索引与余弦检索
│   ├── build.py              # python -m rag.build 建索引
│   ├── generate.py           # DeepSeek 生成
│   └── qa.py                 # 问答编排与降级
├── tests/                    # pytest 纯函数测试
├── .github/workflows/ci.yml  # CI：编译 + 测试
└── docs/                     # 界面截图
    ├── ui-home.png
    ├── detect.png
    └── stats-knowledge.png
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 目标检测 | YOLOv8 (Ultralytics) |
| 前端界面 | Streamlit |
| 检索 | bge-small-zh-v1.5 + numpy 余弦相似度 |
| 生成 | DeepSeek `deepseek-chat`（OpenAI 兼容接口） |
| 图像处理 | OpenCV / Pillow |
| 部署 | Docker |

## 已知局限

- 知识库只有 6 类缺陷、每类一篇文档，超出范围的提问会回退到规则或直接说资料不足
- 检索质量取决于文档质量：块是按标题与段落机选的固定 420 字，没有做语义分段与重排
- 检索只按相似度取前四、没有按类别过滤，实测中「划痕」的问题会带进一段「压入氧化皮」的内容；回答里用得上的是同类段落，但引用区会多出一条无关来源
- 生成结果依赖外部模型，可能仍出现与资料不符的表述；引用只标注到段落，没有逐句核对
- 只做单图推理，没有视频流、批量目录或产线时序的接入
- 权重、索引与训练产物都不入库，仓库不能开箱即用地检测
- 指标只有 mAP，没有误检/漏检的业务成本评估；crazing 一类的 mAP50 只有 0.441，细长、低对比度缺陷是弱项

## License

MIT，见 [LICENSE](LICENSE)。
