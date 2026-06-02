# 可解释的谣言检测

《人工智能导论》课程大作业项目。目标是基于推文数据集构建一个可解释的谣言检测系统：输入一条英文推文，输出二分类检测结果，并生成中文判断依据。

## 任务目标

系统需要实现以下基本功能：

- 输入：一条推文文本。
- 输出 1：二分类标签，`0` 表示非谣言，`1` 表示谣言。
- 输出 2：一段中文判断依据，解释为什么判定为谣言或非谣言。
- 要求模型在 `val.csv` 上具有较好的准确率，并保持合理运行时间。

## 项目结构

```text
rumor-detector/
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── train.csv
│   └── val.csv
├── src/
│   ├── train_baseline.py
│   ├── train_bigru.py
│   ├── train_bert.py
│   ├── evaluate.py
│   ├── llm_api.py
│   ├── prompts.py
│   ├── explain.py
│   ├── rag.py
│   └── inference.py
├── models/
│   ├── lr_model.pkl
│   ├── bigru.pt
│   └── bert_model/
└── outputs/
    └── figures/
```

当前已创建基础目录，后续按上面的文件名逐步补充代码和文档。

## 四人分工

| 成员 | 角色 | 负责内容 | 主要产出 |
|---|---|---|---|
| A | 组长 / 系统集成 / 文档 | 建立 GitHub 仓库，整合分类模型和解释模型，维护 README，EDA 与报告图表绘制，统稿最终报告 | `README.md`、`src/inference.py`、最终 `report.pdf` |
| B | 分类模型负责人 | TF-IDF + LR 基线、BiGRU、BERT微调，保存最优模型 | `src/train_baseline.py`、`src/train_bigru.py`、`src/train_bert.py`、`models/` |
| C | 解释模型负责人 | 对接 SJTU LLM API，设计 Prompt，生成判断依据  | `src/llm_api.py`、`src/prompts.py`、`src/explain.py` |
| D | 实验评估 / 可视化 / 报告实验 | 统一评估各模型，统计 accuracy、precision、recall、F1，绘制混淆矩阵和对比图，分析解释质量 | `src/evaluate.py`、`outputs/figures/`、实验结果表格 |

协作要求：

- 每位成员都应通过自己的分支提交代码或文档，保留 GitHub 贡献记录。
- 建议分支命名：`feature/baseline`、`feature/bert`、`feature/explain`、`feature/evaluate`。
- 每次提交尽量只包含一个清晰任务，例如“add tfidf baseline”或“add llm prompt template”。

## 环境配置

推荐使用 Conda 创建独立环境。仓库同时提供 `environment.yml` 和 `requirements.txt`，助教可以任选一种方式配置。

方式一：使用 Conda 环境文件，推荐优先使用。

```bash
conda env create -f environment.yml
conda activate rumor-detector
```

方式二：手动创建 Conda 环境后使用 pip 安装。

```bash
conda create -n rumor-detector python=3.11 -y
conda activate rumor-detector
pip install -r requirements.txt
```

如果需要使用 GPU 训练 BERT，请根据本机 CUDA 版本安装 PyTorch。可参考 PyTorch 官网选择对应命令。若只是运行逻辑回归基线和单条推理，CPU 环境即可。

```bash
pip install torch torchvision torchaudio
```

最低依赖包括：

- `pandas`：读取 CSV 数据。
- `scikit-learn`：TF-IDF、逻辑回归、评估指标。
- `torch`：BiGRU 和深度学习模型训练。
- `transformers`：BERT微调。
- `matplotlib`、`seaborn`：实验结果可视化。
- `joblib`：保存和加载传统机器学习模型。
- `requests`：调用 SJTU LLM API。

## 数据说明

数据文件位于：

```text
data/train.csv
data/val.csv
```

字段说明：

| 字段 | 含义 |
|---|---|
| `id` | 推文唯一 ID |
| `text` | 推文内容 |
| `label` | 标签，`0` 为非谣言，`1` 为谣言 |
| `event` | 事件主题类别 |

## 成员分工流程

**依赖关系：** A 依赖 B（模型文件）和 C（`explain.py` 的 `generate_explanation()` 接口）；D 依赖 B（模型文件）；C 独立开发。

**1. 成员 B — 分类模型训练**

在 `src/` 下完成三个训练脚本，训练结果保存至 `models/`：

| 脚本 | 产出 | 说明 |
|------|------|------|
| `train_baseline.py` | `models/lr_model.pkl` | TF-IDF + 逻辑回归，快速基线 |
| `train_bigru.py` | `models/bigru.pt` | 双向 GRU + 词嵌入，深度学习基线 |
| `train_bert.py` | `models/bert_model/` | BERT 微调，主力模型 |

模型文件是后续 A（推理）和 D（评估）的共同依赖，需最先完成。

**2. 成员 C — 解释生成模块**

在 `src/` 下完成三层解释管线：

| 文件 | 职责 |
|------|------|
| `llm_api.py` | 封装 SJTU LLM API 调用，支持超时重试 |
| `prompts.py` | 管理 Prompt 模板（v0 基础版 / v1 思维链引导 / fewshot 少样本） |
| `explain.py` | 接收文本 + 分类标签 + 置信度，调用 LLM 生成中文判断依据；LLM 不可用时自动回退到规则模板 |

对外暴露 `generate_explanation(text, label, confidence) -> str`，供 A 的 `inference.py` 调用。C 不依赖其他人，可并行开发。

**3. 成员 D — 模型评估**

编写 `src/evaluate.py`，在验证集 `val.csv` 上对三个模型（LR / BiGRU / BERT）统一评估，输出 accuracy、precision、recall、F1 并汇总对比。可选：绘制混淆矩阵和对比柱状图至 `outputs/figures/`。

**4. 成员 A — 系统集成与报告**

编写 `src/inference.py`，实现 `RumourDetectClass`：

- 加载 B 的模型做分类 → 得到 label + confidence
- 调用 C 的 `generate_explanation()` 生成中文判断依据
- 提供命令行入口：`python src/inference.py -t "推文内容"`

最后统稿 `report.pdf`（≤2000 字），维护 README。A 的工作在 B 和 C 完成后收尾。


## 常用命令

训练逻辑回归基线：

```bash
python src/train_baseline.py
```

训练 BiGRU：

```bash
python src/train_bigru.py
```

训练 BERT：

```bash
python src/train_bert.py
```

评估模型（三模型对比，输出 accuracy、precision、recall、F1）：

```bash
python src/evaluate.py
```

单条推理（分类 + LLM 中文解释）：

```bash
python src/inference.py --text "Breaking news: example tweet text"
```

其他推理选项：

```bash
# 只用 LR / BiGRU 模型
python src/inference.py --text "xxx" --model lr
python src/inference.py --text "xxx" --model bigru

# 仅分类，跳过 LLM 解释（速度快）
python src/inference.py --text "xxx" --no-explain
```

解释效果抽查（随机抽取验证集样本展示分类 + 解释）：

```bash
python test_explain.py -n 10
```

## 最终提交内容

组长最终在 Canvas 提交 GitHub 仓库地址。仓库中应至少包含：

- `README.md`：项目说明、环境配置、运行方法。
- `report.pdf`：大作业报告，不超过 2000 字。
- `src/`：模型训练、评估、推理和解释代码。
- `data/`：课程提供的数据集。
- `models/`：可运行推理所需的模型文件或模型下载说明。
- `outputs/`：预测结果、图表或实验记录。

## 注意事项

- 如果使用大语言模型生成判断依据，应使用学校提供的 SJTU API，方便助教复现。
- Prompt 中不要让模型编造外部事实，解释应尽量引用推文中的具体内容。
- 最终系统必须能做到：输入一条文本，输出分类结果和判断依据。
- 报告篇幅有限，重点写方法、实验结果、解释设计和小组分工。
