"""
统一模型评估（成员 D）
===================================================================
在验证集 val_clean.csv 上评估两个模型（LR / BiGRU），输出：

  1. 指标表        outputs/metrics.csv  +  outputs/metrics.md
                   （accuracy / precision / recall / F1，宏平均 + 谣言类）
  2. 混淆矩阵图    outputs/figures/confusion_<model>.png
  3. 对比柱状图    outputs/figures/metrics_compare.png
  4. 错误样本      outputs/errors_<model>.csv（被预测错的验证集样本）

设计要点（保证与 B 的模型“一步对接”）：
  - 缺失的模型会被自动跳过，不会让整个评估失败；
  - BiGRU 支持两种 checkpoint 格式：含 vocab 的 dict，或仅 state_dict
    的旧格式（后者从 train_clean.csv 确定性重建词表）。

用法：
    python src/evaluate.py                 # 评估所有可用模型
    python src/evaluate.py --models lr bigru
"""

import argparse
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")

sns.set_theme(style="whitegrid")
plt.rcParams["axes.unicode_minus"] = False

# 选择一个本机真实安装的中文字体；都没有则退化为英文标签（数值不受影响）。
from matplotlib import font_manager  # noqa: E402

_installed = {f.name for f in font_manager.fontManager.ttflist}
_cjk_font = next(
    (f for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "WenQuanYi Zen Hei", "Arial Unicode MS", "PingFang SC"]
     if f in _installed),
    None,
)
if _cjk_font:
    plt.rcParams["font.sans-serif"] = [_cjk_font]
    LABEL_NAMES = ["非谣言 (0)", "谣言 (1)"]
    _T = {"cm": "混淆矩阵", "xpred": "预测标签", "ytrue": "真实标签",
          "score": "分数", "compare": "两模型在验证集上的指标对比"}
else:  # 无中文字体（如部分 Linux 环境）→ 全英文，避免乱码方块
    LABEL_NAMES = ["Non-rumor (0)", "Rumor (1)"]
    _T = {"cm": "Confusion Matrix", "xpred": "Predicted", "ytrue": "True",
          "score": "Score", "compare": "Model comparison on validation set"}


# ═══════════════════════════════════════════════════════════════════
# 各模型的预测函数：输入文本列表，返回预测标签 np.ndarray
# 不可用时返回 None（自动跳过）
# ═══════════════════════════════════════════════════════════════════

def predict_lr(texts):
    path = os.path.join(MODELS_DIR, "lr_model.pkl")
    if not os.path.exists(path):
        return None
    data = joblib.load(path)
    model, vectorizer = data["model"], data["vectorizer"]
    clean = [re.sub(r"[^\w\s]", "", str(t).lower()) for t in texts]
    return model.predict(vectorizer.transform(clean))


def predict_bigru(texts):
    path = os.path.join(MODELS_DIR, "bigru.pt")
    if not os.path.exists(path):
        return None
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from text_utils import MAX_LEN, build_vocab, encode
    from inference import BiGRUInference

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # ── 新格式：复用 BiGRUInference.from_checkpoint ──
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model, vocab, max_len = BiGRUInference.from_checkpoint(path, device="cpu")
    else:
        # ── 旧格式兼容（仅 state_dict，从训练集重建词表） ──
        import torch.nn as nn

        state = ckpt
        train_df = pd.read_csv(os.path.join(DATA_DIR, "train_clean.csv"))
        vocab = build_vocab(train_df["text"])
        emb_dim = state["embedding.weight"].shape[1]
        hidden_dim = state["bigru.weight_hh_l0"].shape[1]
        vocab_size = state["embedding.weight"].shape[0]
        max_len = MAX_LEN

        class _BiGRULegacy(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                self.embed_dropout = nn.Dropout(0.0)
                self.bigru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.gru_dropout = nn.Dropout(0.0)
                self.fc = nn.Linear(hidden_dim * 2, 1)

            def forward(self, x):
                emb = self.embedding(x)
                emb = self.embed_dropout(emb)
                _, h = self.bigru(emb)
                h = torch.cat([h[0], h[1]], dim=1)
                h = self.gru_dropout(h)
                return self.fc(h).squeeze(1)

        model = _BiGRULegacy()
        model.load_state_dict(state)
        model.eval()

    ids = torch.tensor([encode(t, vocab, max_len) for t in texts], dtype=torch.long)
    preds = []
    with torch.no_grad():
        for i in range(0, len(ids), 64):
            logits = model(ids[i : i + 64])
            preds.append((torch.sigmoid(logits) > 0.5).int().numpy())
    return np.concatenate(preds)


def predict_bert(texts):
    path = os.path.join(MODELS_DIR, "bert_model")
    if not os.path.isdir(path) or not os.listdir(path):
        return None
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()

    texts = [str(t) for t in texts]
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = tokenizer(
                texts[i : i + 32], truncation=True, padding=True,
                max_length=64, return_tensors="pt",
            )
            logits = model(**batch).logits
            preds.append(logits.argmax(dim=-1).numpy())
    return np.concatenate(preds)


PREDICTORS = {"lr": predict_lr, "bigru": predict_bigru, "bert": predict_bert}
DISPLAY_NAME = {"lr": "TF-IDF + LR", "bigru": "BiGRU", "bert": "BERT"}


# ═══════════════════════════════════════════════════════════════════
# 指标计算 + 产物生成
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        # 谣言类（正类 = 1）单独看，业务上更关注
        "precision_rumor": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_rumor": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_rumor": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def plot_confusion(y_true, y_pred, key):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_xlabel(_T["xpred"])
    ax.set_ylabel(_T["ytrue"])
    ax.set_title(f"{_T['cm']} — {DISPLAY_NAME[key]}")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"confusion_{key}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_comparison(results):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["准确率 (Accuracy)", "精确率 (Precision)", "召回率 (Recall)", "F1 值 (F1)"]
    keys = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / max(len(keys), 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, key in enumerate(keys):
        vals = [results[key][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=DISPLAY_NAME[key])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width * (len(keys) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(_T["score"])
    ax.set_title(_T["compare"])
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "metrics_compare.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def df_to_markdown(table: pd.DataFrame) -> str:
    """把 DataFrame 转成 Markdown 表格（不依赖 tabulate，避免额外依赖）。"""
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_error_samples(df, y_true, y_pred, key):
    mask = np.array(y_true) != np.array(y_pred)
    err = df.loc[mask, ["id", "text", "label", "event"]].copy()
    err["pred"] = np.array(y_pred)[mask]
    out = os.path.join(OUT_DIR, f"errors_{key}.csv")
    err.to_csv(out, index=False, encoding="utf-8-sig")
    return out, int(mask.sum())


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lr", "bigru"],
                        choices=["lr", "bigru", "bert"])
    parser.add_argument("--val", default=os.path.join(DATA_DIR, "val_clean.csv"))
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    val_df = pd.read_csv(args.val)
    texts = val_df["text"].astype(str).tolist()
    y_true = val_df["label"].tolist()
    print(f"验证集: {args.val}  ({len(val_df)} 条, 标签分布 {val_df['label'].value_counts().sort_index().to_dict()})\n")

    results = {}
    for key in args.models:
        print(f">>> 评估 {DISPLAY_NAME[key]} ...")
        try:
            y_pred = PREDICTORS[key](texts)
        except Exception as e:
            print(f"    [跳过] {DISPLAY_NAME[key]} 评估出错: {e}\n")
            continue
        if y_pred is None:
            print(f"    [跳过] 未找到 {DISPLAY_NAME[key]} 的模型文件\n")
            continue

        m = compute_metrics(y_true, y_pred)
        results[key] = m
        cm_path = plot_confusion(y_true, y_pred, key)
        err_path, n_err = save_error_samples(val_df, y_true, y_pred, key)
        print(f"    准确率={m['accuracy']:.4f}  "
              f"精确率(宏平均)={m['precision_macro']:.4f}  "
              f"召回率(宏平均)={m['recall_macro']:.4f}  "
              f"F1(宏平均)={m['f1_macro']:.4f}  "
              f"F1(谣言)={m['f1_rumor']:.4f}")
        print(f"    混淆矩阵 -> {os.path.relpath(cm_path, PROJECT_ROOT)}")
        print(f"    错误样本 {n_err} 条 -> {os.path.relpath(err_path, PROJECT_ROOT)}\n")

    if not results:
        print("没有可评估的模型。请先训练模型（train_baseline / train_bigru / train_bert）。")
        return

    # ── 汇总指标表 ──
    # 指标中英文映射
    _METRIC_CN = {
        "accuracy": "准确率",
        "precision_macro": "精确率（宏平均）",
        "recall_macro": "召回率（宏平均）",
        "f1_macro": "F1（宏平均）",
        "precision_rumor": "精确率（谣言类）",
        "recall_rumor": "召回率（谣言类）",
        "f1_rumor": "F1（谣言类）",
    }
    rows = []
    for key, m in results.items():
        row = {"模型": DISPLAY_NAME[key]}
        row.update({_METRIC_CN[k]: round(v, 4) for k, v in m.items()})
        rows.append(row)
    table = pd.DataFrame(rows)

    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_path = os.path.join(OUT_DIR, "metrics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 验证集评估结果\n\n")
        f.write(f"验证集：`{os.path.relpath(args.val, PROJECT_ROOT)}`，共 {len(val_df)} 条。\n\n")
        f.write(df_to_markdown(table))
        f.write("\n\n")
        f.write("- `*_macro`：两类的宏平均；`*_rumor`：谣言类（标签=1）单独指标。\n")
        f.write("- **准确率 (Accuracy)**：所有样本中预测正确的比例。\n")
        f.write("- **精确率 (Precision)**：预测为某类的样本中，真正属于该类的比例（查准率）。\n")
        f.write("- **召回率 (Recall)**：某类真实样本中，被正确识别出来的比例（查全率）。\n")
        f.write("- **F1 值**：精确率与召回率的调和平均，综合衡量模型性能。\n")
    cmp_path = plot_comparison(results)

    print("=" * 60)
    print("汇总指标：")
    print(table.to_string(index=False))
    print("=" * 60)
    print(f"指标表 -> {os.path.relpath(csv_path, PROJECT_ROOT)} / {os.path.relpath(md_path, PROJECT_ROOT)}")
    print(f"对比图 -> {os.path.relpath(cmp_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
