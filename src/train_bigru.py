"""
BiGRU + 词嵌入 深度学习基线训练
- 读取 train_clean.csv 训练，val_clean.csv 自评
- 保存 models/bigru.pt

保存格式（dict）：
    {
        'model_state': state_dict,
        'vocab': {...},
        'config': {'embedding_dim':.., 'hidden_dim':.., 'max_len':.., 'dropout':..}
    }
evaluate.py 兼容两种格式：上面的 dict，或仅 state_dict 的旧格式
（旧格式会自动从 train_clean.csv 重建词表）。
"""

import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_utils import MAX_LEN, build_vocab, encode  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RumorDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(
            encode(self.texts[idx], self.vocab, self.max_len), dtype=torch.long
        )
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y


class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.bigru = nn.GRU(
            embedding_dim, hidden_dim,
            batch_first=True, bidirectional=True,
        )
        self.gru_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)
        _, h = self.bigru(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        h = self.gru_dropout(h)
        return self.fc(h).squeeze(1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.BCEWithLogitsLoss()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * y.size(0)
    return correct / total, total_loss / total


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="BiGRU 谣言检测训练")
    parser.add_argument("--embedding_dim", type=int, default=200,
                        help="词嵌入维度（默认 200）")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="GRU 隐层维度（默认 256）")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout 概率（默认 0.3）")
    parser.add_argument("--epochs", type=int, default=15,
                        help="最大训练轮数（默认 15）")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小（默认 32）")
    parser.add_argument("--max_len", type=int, default=MAX_LEN,
                        help=f"最大 token 长度（默认 {MAX_LEN}）")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率（默认 1e-3）")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Adam 权重衰减（默认 1e-4）")
    parser.add_argument("--min_freq", type=int, default=2,
                        help="词表最小词频（默认 2）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train_clean.csv"))
    val_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "val_clean.csv"))

    vocab = build_vocab(train_df["text"], min_freq=args.min_freq)
    print(f"训练集: {len(train_df)} 条  验证集: {len(val_df)} 条")
    print(f"词表大小: {len(vocab)}  设备: {DEVICE}")
    print(f"超参: emb={args.embedding_dim}  hidden={args.hidden_dim}  "
          f"dropout={args.dropout}  epochs={args.epochs}")

    train_loader = DataLoader(
        RumorDataset(train_df, vocab, args.max_len),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        RumorDataset(val_df, vocab, args.max_len),
        batch_size=args.batch_size,
    )

    model = BiGRU(
        len(vocab), args.embedding_dim, args.hidden_dim, args.dropout,
    ).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    best_epoch = 0
    out_path = os.path.join(PROJECT_ROOT, "models", "bigru.pt")
    print(f"\n{'=' * 55}")
    print(f"  {'Epoch':<7} {'Train Acc':<11} {'Train Loss':<11} {'Val Acc':<9} {'Val Loss':<9}")
    print(f"{'=' * 55}")

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        train_acc, train_loss = evaluate(model, train_loader)
        val_acc, val_loss = evaluate(model, val_loader)
        scheduler.step(val_acc)

        improved = ""
        if val_acc >= best_acc:
            if val_acc > best_acc:
                improved = "  ← 最优"
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "embedding_dim": args.embedding_dim,
                        "hidden_dim": args.hidden_dim,
                        "max_len": args.max_len,
                        "dropout": args.dropout,
                    },
                },
                out_path,
            )

        print(f"  {epoch + 1:<7} {train_acc:<11.4f} {train_loss:<11.4f} "
              f"{val_acc:<9.4f} {val_loss:<9.4f}{improved}")

    print(f"{'=' * 55}")
    print(f"最优 Val Acc: {best_acc:.4f} (epoch {best_epoch})，模型已保存为 {out_path}")


if __name__ == "__main__":
    main()
