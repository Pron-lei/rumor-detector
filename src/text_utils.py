"""
BiGRU 共享文本工具：分词 / 构建词表 / 编码。

train_bigru.py 和 evaluate.py 都从这里导入，保证训练与评估时
的分词逻辑、词表构建方式完全一致——这是 BiGRU 能正确评估的前提。

词表由 train_clean.csv 确定性地重建（相同数据 + 相同 min_freq → 相同词表），
因此即使 checkpoint 里没有保存 vocab，评估端也能复现出同一份词表。
"""

import re
from collections import Counter

MAX_LEN = 64
MIN_FREQ = 2


def tokenize(text: str):
    return re.findall(r"\w+", str(text).lower())


def build_vocab(texts, min_freq: int = MIN_FREQ) -> dict:
    """从文本序列构建词表，{'<PAD>':0, '<UNK>':1, ...}。"""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def encode(text: str, vocab: dict, max_len: int = MAX_LEN):
    """文本 → 定长 id 列表（截断 / <PAD> 补齐）。"""
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids
