"""
BiGRU + 词嵌入 深度学习基线训练
- 读取 train_clean.csv 训练，val_clean.csv 自评
- 保存 models/bigru.pt

保存格式（dict）：
    {
        'model_state': state_dict,
        'vocab': {...},
        'config': {'embedding_dim':.., 'hidden_dim':.., 'max_len':..}
    }
evaluate.py 兼容两种格式：上面的 dict，或仅 state_dict 的旧格式
（旧格式会自动从 train_clean.csv 重建词表）。
"""

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

BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
EPOCHS = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RumorDataset(Dataset):
    def __init__(self, df, vocab):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y


class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bigru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.bigru(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = (torch.sigmoid(model(x)) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(42)

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train_clean.csv"))
    val_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "val_clean.csv"))

    vocab = build_vocab(train_df["text"])
    print(f"词表大小: {len(vocab)}")

    train_loader = DataLoader(RumorDataset(train_df, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RumorDataset(val_df, vocab), batch_size=BATCH_SIZE)

    model = BiGRU(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    out_path = os.path.join(PROJECT_ROOT, "models", "bigru.pt")
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}, Val Acc: {acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "embedding_dim": EMBEDDING_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "max_len": MAX_LEN,
                    },
                },
                out_path,
            )
    print(f"最优 Val Acc: {best_acc:.4f}，模型已保存为 {out_path}")


if __name__ == "__main__":
    main()
