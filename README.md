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
├── report.md
├── requirements.txt
├── environment.yml
├── .env.example
├── .gitignore
├── test_explain.py                  # 解释效果抽查脚本
├── data/
│   ├── train.csv                    # 原始训练集
│   ├── val.csv                      # 原始验证集
│   ├── train_clean.csv              # 清洗后训练集
│   ├── val_clean.csv                # 清洗后验证集
│   └── clean_report.txt             # 数据清洗报告
├── src/
│   ├── data_clean.py                # 数据清洗
│   ├── text_utils.py                # BiGRU 分词/词表工具
│   ├── train_baseline.py            # TF-IDF + 逻辑回归训练
│   ├── train_bigru.py               # BiGRU 训练
│   ├── train_bert.py                # BERT 微调训练（可选加分项）
│   ├── evaluate.py                  # 统一评估
│   ├── llm_api.py                   # SJTU LLM API 封装
│   ├── prompts.py                   # Prompt 模板管理
│   ├── explain.py                   # 解释生成模块
│   └── inference.py                 # 端到端推理入口
├── models/
│   ├── lr_model.pkl                 # 逻辑回归模型
│   ├── bigru.pt                     # BiGRU 模型（权重 + 词表 + 配置）
│   └── bert_model/                  # BERT 模型目录（需手动训练）
└── outputs/
    ├── metrics.csv                  # 评估指标表
    ├── metrics.md                   # 评估指标表（Markdown）
    ├── errors_lr.csv                # LR 错误样本
    ├── errors_bigru.csv             # BiGRU 错误样本
    └── figures/
        ├── confusion_lr.png         # LR 混淆矩阵
        ├── confusion_bigru.png      # BiGRU 混淆矩阵
        └── metrics_compare.png      # 模型对比柱状图
```

## 四人分工

| 成员 | 角色 | 负责内容 | 主要产出 |
|---|---|---|---|
| A | 组长 / 系统集成 / 文档 | 建立 GitHub 仓库，整合分类模型和解释模型，维护 README，报告撰写 | `README.md`、`src/inference.py`、`report.md` |
| B | 分类模型负责人 | TF-IDF + LR 基线、BiGRU、BERT 微调，保存模型 | `src/train_baseline.py`、`src/train_bigru.py`、`src/train_bert.py`、`models/` |
| C | 解释模型负责人 | 对接 SJTU LLM API，设计 Prompt，生成判断依据 | `src/llm_api.py`、`src/prompts.py`、`src/explain.py` |
| D | 实验评估 / 可视化 | 统一评估各模型，绘制混淆矩阵和对比图，分析解释质量 | `src/evaluate.py`、`outputs/figures/`、实验结果表格 |

协作要求：

- 每位成员都应通过自己的分支提交代码或文档，保留 GitHub 贡献记录。
- 建议分支命名：`feature/baseline`、`feature/bigru`、`feature/bert`、`feature/explain`、`feature/evaluate`。
- 每次提交尽量只包含一个清晰任务，例如"add tfidf baseline"或"add llm prompt template"。

## 环境配置

> **前置条件**：请确保已安装 Anaconda 或 Miniconda，终端中可执行 `conda` 命令。打开终端后默认处于 `base` 环境是正常的。

**第一步：进入项目根目录。**

```bash
cd rumor-detector   # 根据实际路径调整，如 cd "D:\rumor detector"
```

后续所有命令均需在项目根目录下执行。

**第二步：创建并激活环境。**

方式一：使用 Conda 环境文件（推荐）

```bash
conda env create -f environment.yml
conda activate rumor-detector
```

方式二：手动创建 Conda 环境后使用 pip 安装

```bash
conda create -n rumor-detector python=3.11 -y
conda activate rumor-detector
pip install -r requirements.txt
```

如果需要使用 GPU 训练 BERT，请根据本机 CUDA 版本安装 PyTorch。可参考 [PyTorch 官网](https://pytorch.org/) 选择对应命令。若只运行逻辑回归基线和单条推理，CPU 环境即可。

```bash
pip install torch torchvision torchaudio
```

最低依赖包括：

- `pandas`：读取 CSV 数据。
- `scikit-learn`：TF-IDF、逻辑回归、评估指标。
- `torch`：BiGRU 和深度学习模型训练。
- `transformers`：BERT 微调。
- `matplotlib`、`seaborn`：实验结果可视化。
- `joblib`：保存和加载传统机器学习模型。
- `requests`：调用 SJTU LLM API。

## 数据说明

数据文件位于 `data/` 目录：

| 字段 | 含义 |
|---|---|
| `id` | 推文唯一 ID |
| `text` | 推文内容 |
| `label` | 标签，`0` 为非谣言，`1` 为谣言 |
| `event` | 事件主题类别 |

原始数据 `train.csv`（2,840 条）和 `val.csv`（401 条）经过清洗后得到 `train_clean.csv`（2,779 条）和 `val_clean.csv`（400 条）。清洗流程详见 `src/data_clean.py`，清洗报告见 `data/clean_report.txt`。

## 成员分工流程

**依赖关系：** A 依赖 B（模型文件）和 C（`explain.py` 的 `generate_explanation()` 接口）；D 依赖 B（模型文件）；C 独立开发。

**1. 成员 B — 分类模型训练**

在 `src/` 下完成训练脚本，训练结果保存至 `models/`：

| 脚本 | 产出 | 说明 |
|------|------|------|
| `train_baseline.py` | `models/lr_model.pkl` | TF-IDF + 逻辑回归，快速基线 |
| `train_bigru.py` | `models/bigru.pt` | 双向 GRU + 词嵌入，深度学习模型 |
| `train_bert.py` | `models/bert_model/` | BERT 微调，可选加分项（模型 ~420MB，不入库） |

模型文件是后续 A（推理）和 D（评估）的共同依赖，LR 和 BiGRU 需最先完成。BERT 已训练完成但模型文件不入库，如需复现需手动运行训练脚本（见下方常用命令）。

**2. 成员 C — 解释生成模块**

在 `src/` 下完成三层解释管线：

| 文件 | 职责 |
|------|------|
| `llm_api.py` | 封装 SJTU LLM API 调用，支持超时重试（30s / 最多 2 次），内置默认 API Key |
| `prompts.py` | 管理 Prompt 模板（v0 基础版 / v1 思维链引导 / fewshot 少样本 / rag 检索增强） |
| `explain.py` | 接收文本 + 分类标签 + 置信度，调用 LLM 生成中文判断依据；LLM 不可用时自动回退到规则模板 |

对外暴露 `generate_explanation(text, label, confidence) -> str`，供 A 的 `inference.py` 调用。C 不依赖其他人，可并行开发。

**3. 成员 D — 模型评估**

编写 `src/evaluate.py`，在验证集 `val_clean.csv` 上评估 LR 和 BiGRU 两个模型（BERT 需手动指定 `--models lr bigru bert`），输出准确率、精确率、召回率、F1 并汇总对比，同时生成混淆矩阵图和对比柱状图至 `outputs/figures/`。

**4. 成员 A — 系统集成与报告**

编写 `src/inference.py`，实现 `RumourDetectClass`：

- 加载 B 的模型做分类 → 得到 label + confidence
- 调用 C 的 `generate_explanation()` 生成中文判断依据
- 提供命令行入口（单条检测 / 交互模式），默认使用 BiGRU 模型
- 编写 `report.md`，统稿最终报告

A 的工作在 B 和 C 完成后收尾。

## 常用命令

> 以下命令均需在项目根目录下、`rumor-detector` 环境已激活的状态运行。

### 一键运行（助教验收推荐流程）

```bash
# 1. 数据清洗（如尚未执行）
python src/data_clean.py

# 2. 评估两个默认模型（LR + BiGRU）
python src/evaluate.py

# 3. 单条推理测试（默认 BiGRU）
python src/inference.py --text "Breaking news: massive earthquake hits the city!"
```

### 训练模型

```bash
# 逻辑回归基线（秒级完成）
python src/train_baseline.py

# BiGRU（CPU 约 2-3 分钟）
python src/train_bigru.py

# BERT（可选加分项，CPU 约 20-30 分钟，建议 GPU）
# 训练完成后模型保存在 models/bert_model/，约 420MB
python src/train_bert.py --epochs 3 --batch_size 16
```

### 评估模型

```bash
# 评估默认两模型（LR + BiGRU）
python src/evaluate.py

# 训练 BERT 后，加入 BERT 一起评估
python src/evaluate.py --models lr bigru bert
```

### 推理

```bash
# 默认 BiGRU 模型，单条检测
python src/inference.py --text "Breaking news: example tweet text"

# 切换模型
python src/inference.py --text "xxx" --model lr
python src/inference.py --text "xxx" --model bigru
python src/inference.py --text "xxx" --model bert    # 需先训练 BERT

# 仅分类，跳过 LLM 解释（速度快）
python src/inference.py --text "xxx" --no-explain

# 交互模式（逐条输入推文检测）
python src/inference.py --interactive
```

### 解释效果抽查

```bash
# 使用规则模板（默认，无需 API）
python test_explain.py -n 10

# 使用 LLM 生成解释
python test_explain.py -n 5 --llm
```

### 关于 BERT 模型

BERT（`bert-base-uncased`）作为可选加分模型，微调后权重约 420MB，**不纳入 Git 仓库**（已通过 `.gitignore` 排除）。助教如需复现 BERT：

```bash
# 训练（需 GPU 或耐心等待 CPU 完成）
python src/train_bert.py --epochs 3 --batch_size 16

# 训练完成后即可在评估和推理中使用
python src/evaluate.py --models lr bigru bert
python src/inference.py --text "xxx" --model bert
```

`outputs/` 中已保存它在验证集上的历史评估结果供参考。

## 最终提交内容

组长最终在 Canvas 提交 GitHub 仓库地址。仓库中应至少包含：

- `README.md`：项目说明、环境配置、运行方法。
- `report.pdf`：大作业报告，不超过 2,000 字。
- `src/`：模型训练、评估、推理和解释代码。
- `data/`：课程提供的数据集及清洗后数据。
- `models/`：可运行推理所需的 LR 和 BiGRU 模型文件。
- `outputs/`：评估指标、图表和错误样本。

## 注意事项

- 如果使用大语言模型生成判断依据，应使用学校提供的 SJTU API，方便助教复现。
- Prompt 中不要让模型编造外部事实，解释应尽量引用推文中的具体内容。
- 最终系统必须能做到：输入一条文本，输出分类结果和判断依据。
- 报告篇幅有限，重点写方法、实验结果、解释设计和小组分工。
- 仓库中 `report.md` 为报告源文件，最终需转为 `report.pdf` 提交。
