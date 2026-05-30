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
├── data/
│   ├── train.csv
│   └── val.csv
├── src/
│   ├── preprocess.py
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
│   └── bert_model/
├── outputs/
│   ├── predictions.csv
│   └── figures/
├── notebooks/
│   ├── eda.ipynb
│   └── result_analysis.ipynb
└── docs/
    └── API使用说明.md
```

当前已创建基础目录，后续按上面的文件名逐步补充代码和文档。

## 四人分工

| 成员 | 角色 | 负责内容 | 主要产出 |
|---|---|---|---|
| A | 组长 / 系统集成 / 文档 | 建立 GitHub 仓库，制定进度，整合分类模型和解释模型，维护 README，统稿最终报告 | `README.md`、`src/inference.py`、最终 `report.pdf` |
| B | 分类模型负责人 | 数据清洗、EDA、TF-IDF + LR 基线、BiGRU、BERT/RoBERTa 微调，保存最优模型 | `src/preprocess.py`、`src/train_baseline.py`、`src/train_bigru.py`、`src/train_bert.py`、`models/` |
| C | 解释模型负责人 | 对接 SJTU LLM API，设计 Prompt，生成判断依据，可选实现 RAG 检索增强 | `src/llm_api.py`、`src/prompts.py`、`src/explain.py`、`src/rag.py` |
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
- `transformers`：BERT / RoBERTa 微调。
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

## 推荐开发流程

1. B 先完成 `TF-IDF + LogisticRegression` 基线模型，保存到 `models/lr_model.pkl`。
2. D 编写统一评估脚本，在 `val.csv` 上输出 accuracy、precision、recall、F1。
3. A 编写 `RumourDetectClass`，打通单条文本推理流程。
4. C 接入 SJTU LLM API，根据文本、分类标签和置信度生成中文解释。
5. B 继续训练 BiGRU 和 BERT，替换基线模型，提升准确率。
6. D 汇总不同模型实验结果，生成图表并撰写实验分析。
7. A 整理 README、报告和最终提交文件。

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

评估模型：

```bash
python src/evaluate.py
```

单条推理：

```bash
python src/inference.py --text "Breaking news: example tweet text"
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
