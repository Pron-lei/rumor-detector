"""
BERT 微调 谣言检测（主力模型）
- 读取 train_clean.csv 训练，val_clean.csv 自评
- 保存到 models/bert_model/（含 tokenizer 与权重，evaluate.py 直接加载）

默认使用 bert-base-uncased；可用 --model_name 指定更小的模型
（如 distilbert-base-uncased / prajjwal1/bert-tiny）以加速 CPU 训练。

针对小数据集已做优化：
  - 默认 dropout=0.3 防止过拟合
  - 默认 lr=5e-5 + weight_decay=0.05
  - 每 epoch 打印 train/val 双指标，方便观察过拟合趋势
  - 仅保存验证集最优模型
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        correct += (out.logits.argmax(dim=-1) == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
        total_loss += out.loss.item() * batch["labels"].size(0)
    return correct / total, total_loss / total


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="BERT 谣言检测训练")
    parser.add_argument("--model_name", default="bert-base-uncased",
                        help="HuggingFace 模型名（默认 bert-base-uncased）")
    parser.add_argument("--epochs", type=int, default=5,
                        help="最大训练轮数（默认 5）")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小（默认 16）")
    parser.add_argument("--max_len", type=int, default=64,
                        help="最大 token 长度（默认 64）")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="学习率（默认 5e-5）")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="分类器与隐层 dropout 概率（默认 0.3，防过拟合）")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="AdamW 权重衰减（默认 0.05）")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="学习率预热比例（默认 0.1）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train_clean.csv"))
    val_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "val_clean.csv"))

    print(f"训练集: {len(train_df)} 条  验证集: {len(val_df)} 条")
    print(f"设备: {DEVICE}")
    print(f"模型: {args.model_name}")
    print(f"超参: epochs={args.epochs}  batch={args.batch_size}  max_len={args.max_len}")
    print(f"      lr={args.lr}  dropout={args.dropout}  wd={args.weight_decay}")

    # ── 加载模型 ──
    print(f"\n加载 {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)

    # 设置 dropout（bert-base 用 hidden_dropout_prob，distilbert 用 classifier_dropout）
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = args.dropout
        config.attention_probs_dropout_prob = args.dropout / 2
    if hasattr(config, "classifier_dropout"):
        config.classifier_dropout = args.dropout

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config, ignore_mismatched_sizes=True,
    ).to(DEVICE)

    # ── 数据加载 ──
    train_loader = DataLoader(
        BertDataset(train_df["text"].astype(str), train_df["label"], tokenizer, args.max_len),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        BertDataset(val_df["text"].astype(str), val_df["label"], tokenizer, args.max_len),
        batch_size=args.batch_size * 2,
    )

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── 训练 ──
    out_dir = os.path.join(PROJECT_ROOT, "models", "bert_model")
    best_acc = 0.0
    best_epoch = 0
    print(f"\n{'=' * 55}")
    print(f"  {'Epoch':<7} {'Train Acc':<11} {'Train Loss':<11} {'Val Acc':<9} {'Val Loss':<9}")
    print(f"{'=' * 55}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += out.loss.item()

        # Epoch 结束 — 评估
        train_acc, train_loss = evaluate(model, train_loader)
        val_acc, val_loss = evaluate(model, val_loader)

        improved = ""
        if val_acc >= best_acc:
            if val_acc > best_acc:
                improved = "  ← 最优"
            best_acc = val_acc
            best_epoch = epoch + 1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

        print(f"  {epoch + 1:<7} {train_acc:<11.4f} {train_loss:<11.4f} "
              f"{val_acc:<9.4f} {val_loss:<9.4f}{improved}")

    print(f"{'=' * 55}")
    print(f"最优 Val Acc: {best_acc:.4f} (epoch {best_epoch})，模型已保存至 {out_dir}")


if __name__ == "__main__":
    main()
