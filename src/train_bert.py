"""
BERT 微调 谣言检测（主力模型）
- 读取 train_clean.csv 训练，val_clean.csv 自评
- 保存到 models/bert_model/（含 tokenizer 与权重，evaluate.py 直接加载）

默认使用 bert-base-uncased；可用 --model_name 指定更小的模型
（如 distilbert-base-uncased / prajjwal1/bert-tiny）以加速 CPU 训练。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

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
    correct, total = 0, 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / total


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train_clean.csv"))
    val_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "val_clean.csv"))

    print(f"加载 {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(DEVICE)

    train_loader = DataLoader(
        BertDataset(train_df["text"].astype(str), train_df["label"], tokenizer, args.max_len),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        BertDataset(val_df["text"].astype(str), val_df["label"], tokenizer, args.max_len),
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    out_dir = os.path.join(PROJECT_ROOT, "models", "bert_model")
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += out.loss.item()
            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch + 1} step {i + 1}/{len(train_loader)} loss {running / 50:.4f}")
                running = 0.0
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}, Val Acc: {acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    print(f"最优 Val Acc: {best_acc:.4f}，模型已保存至 {out_dir}")


if __name__ == "__main__":
    main()
