"""
数据清洗脚本
- 分析并清洗 train.csv / val.csv 中的噪声
- 输出 train_clean.csv / val_clean.csv + 清洗报告
- 原文件不做任何修改
"""

import html
import os
import re
import sys
from collections import Counter

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════
# 单条文本清洗（步骤 1-5）
# 这个函数也可用于推理阶段的新文本清洗
# ═══════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """清洗单条推文文本，返回干净字符串。"""
    if not isinstance(text, str):
        return ""

    # 1. 解码 HTML 实体  &amp; → &  &gt; → >  &#039; → '
    text = html.unescape(text)

    # 2. 移除 URL
    text = re.sub(r"https?://\S+", "", text)

    # 3. 规范化 @mention → @USER
    text = re.sub(r"@\w+", "@USER", text)

    # 4. 规范化 #hashtag → 保留文字去掉 #（处理多#如 ##Word）
    text = re.sub(r"#+(\w+)", r"\1", text)
    # 清理残留的孤零零 # 符号（后面没有紧跟单词的 #）
    text = re.sub(r"#", "", text)

    # 5. 规范化空白
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ═══════════════════════════════════════════════════════════════════
# 噪声分析（清洗前/后各调用一次）
# ═══════════════════════════════════════════════════════════════════

def analyze_noise(df: pd.DataFrame, name: str) -> dict:
    """统计数据集噪声情况，返回统计字典。"""
    texts = df["text"].astype(str)
    stats = {
        "name": name,
        "rows": len(df),
        "label_dist": dict(df["label"].value_counts().sort_index()),
        "event_dist": dict(df["event"].value_counts().sort_index()),
        "missing": int(df.isnull().sum().sum()),
        "exact_dup": int(df.duplicated(subset=["text"]).sum()),
        "url_count": int(texts.str.contains(r"https?://\S+", regex=True).sum()),
        "html_count": int(texts.str.contains(r"&[a-z]+;", regex=True).sum()),
        "mention_count": int(texts.str.contains(r"@\w+", regex=True).sum()),
        "hashtag_count": int(texts.str.contains(r"#\w+", regex=True).sum()),
        "min_len": int(texts.str.len().min()),
        "max_len": int(texts.str.len().max()),
        "mean_len": float(texts.str.len().mean()),
    }
    return stats


def print_stats(stats: dict):
    """打印统计信息。"""
    print(f"\n{'='*50}")
    print(f"  {stats['name']}: {stats['rows']} 条")
    print(f"{'='*50}")
    print(f"  标签分布: {stats['label_dist']}")
    print(f"  事件分布: {stats['event_dist']}")
    print(f"  缺失值: {stats['missing']}")
    print(f"  完全重复文本: {stats['exact_dup']}")
    print(f"  含 URL: {stats['url_count']} ({stats['url_count']/max(stats['rows'],1)*100:.1f}%)")
    print(f"  含 HTML 实体: {stats['html_count']}")
    print(f"  含 @mention: {stats['mention_count']}")
    print(f"  含 #hashtag: {stats['hashtag_count']}")
    print(f"  文本长度: min={stats['min_len']}, max={stats['max_len']}, mean={stats['mean_len']:.0f}")


# ═══════════════════════════════════════════════════════════════════
# 去重 + 标签冲突处理（步骤 6-7）
# ═══════════════════════════════════════════════════════════════════

def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    去重流程：
    1. 找到标签冲突的文本（完全相同文本有不同 label），全部删除
    2. 删除完全重复文本（完全相同的 text + label），保留第一条
    3. 删除近重复文本（去除 URL 后相同的 text + label），保留第一条

    返回 (清洗后DataFrame, 统计信息)
    """
    report = {}
    initial = len(df)

    # ── 6a. 查找标签冲突 ──
    conflict_texts = set()
    for text, group in df.groupby("text"):
        if group["label"].nunique() > 1:
            conflict_texts.add(text)

    report["conflict_groups"] = len(conflict_texts)
    report["conflict_rows"] = int(df["text"].isin(conflict_texts).sum())

    if conflict_texts:
        print(f"\n  ⚠ 发现 {len(conflict_texts)} 组标签冲突，删除 {report['conflict_rows']} 条:")
        for t in conflict_texts:
            group = df[df["text"] == t]
            print(f"    labels={list(group['label'])} → \"{t[:80]}...\"")
        df = df[~df["text"].isin(conflict_texts)]

    # ── 6b. 删除完全重复（相同 text + label） ──
    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"], keep="first")
    report["exact_dup_removed"] = before - len(df)

    # ── 7. 删除近重复（临时列: 去除 URL 后的文本） ──
    df = df.copy()
    df["_text_nourl"] = df["text"].apply(lambda t: re.sub(r"https?://\S+", "", str(t)).strip())
    before = len(df)
    df = df.drop_duplicates(subset=["_text_nourl", "label"], keep="first")
    report["near_dup_removed"] = before - len(df)

    df = df.drop(columns=["_text_nourl"])

    report["total_removed"] = initial - len(df)
    report["remaining"] = len(df)
    return df, report


# ═══════════════════════════════════════════════════════════════════
# 训练/验证集重叠移除（步骤 8）
# ═══════════════════════════════════════════════════════════════════

def remove_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """从训练集中删除同时在验证集中出现的文本（大小写不敏感）。"""
    val_texts = set(val_df["text"].str.lower().str.strip())
    overlap_mask = train_df["text"].str.lower().str.strip().isin(val_texts)
    overlap_count = int(overlap_mask.sum())

    if overlap_count > 0:
        print(f"\n  ⚠ 训练/验证集重叠 {overlap_count} 条，从训练集删除:")
        for t in train_df[overlap_mask]["text"]:
            print(f"    \"{t[:80]}...\"")

    return train_df[~overlap_mask], overlap_count


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    # 强制 utf-8 输出（Windows 兼容）
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  谣言检测数据集清洗")
    print("=" * 60)

    # ── 加载原始数据 ──
    train_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
    val_path = os.path.join(PROJECT_ROOT, "data", "val.csv")
    train_raw = pd.read_csv(train_path)
    val_raw = pd.read_csv(val_path)

    # ── 清洗前分析 ──
    print("\n>>> 清洗前数据概览")
    stats_before_train = analyze_noise(train_raw, "训练集 (原始)")
    stats_before_val = analyze_noise(val_raw, "验证集 (原始)")
    print_stats(stats_before_train)
    print_stats(stats_before_val)

    # ── 应用文本清洗（步骤 1-5） ──
    print("\n\n>>> 步骤 1-5: 文本清洗 (URL/HTML/Mention/Hashtag/空白)")
    train_df = train_raw.copy()
    val_df = val_raw.copy()
    train_df["text"] = train_df["text"].apply(clean_text)
    val_df["text"] = val_df["text"].apply(clean_text)

    # ── 去重（步骤 6-7） ──
    print("\n>>> 步骤 6-7: 去重（标签冲突 + 完全重复 + 近重复）")
    train_df, dedup_report = deduplicate(train_df)
    val_df, val_dedup_report = deduplicate(val_df)
    # 合并去重统计（更清晰的分项）
    dedup_total_train = dedup_report["total_removed"]
    dedup_total_val = val_dedup_report["total_removed"]

    # ── 训练/验证集重叠移除（步骤 8） ──
    print("\n>>> 步骤 8: 移除训练/验证集重叠")
    train_df, overlap_count = remove_overlap(train_df, val_df)

    # ── 清洗后分析 ──
    print("\n\n>>> 清洗后数据概览")
    stats_after_train = analyze_noise(train_df, "训练集 (清洗后)")
    stats_after_val = analyze_noise(val_df, "验证集 (清洗后)")
    print_stats(stats_after_train)
    print_stats(stats_after_val)

    # ── 保存清洗后数据 ──
    print("\n\n>>> 保存清洗后数据")
    train_out = os.path.join(PROJECT_ROOT, "data", "train_clean.csv")
    val_out = os.path.join(PROJECT_ROOT, "data", "val_clean.csv")
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    print(f"  训练集: {train_out} ({len(train_df)} 条)")
    print(f"  验证集: {val_out} ({len(val_df)} 条)")

    # ── 生成清洗报告 ──
    report_path = os.path.join(PROJECT_ROOT, "data", "clean_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("数据清洗报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"清洗时间: 自动生成\n")
        f.write(f"清洗脚本: src/data_clean.py\n\n")

        f.write("【清洗前后对比】\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'':>12} {'训练集':>10} {'验证集':>10}\n")
        f.write(f"{'清洗前':>12} {stats_before_train['rows']:>10} {stats_before_val['rows']:>10}\n")
        f.write(f"{'清洗后':>12} {stats_after_train['rows']:>10} {stats_after_val['rows']:>10}\n")
        f.write(f"{'删除':>12} {stats_before_train['rows'] - stats_after_train['rows']:>10} "
                f"{stats_before_val['rows'] - stats_after_val['rows']:>10}\n\n")

        f.write("【训练集 — 删除明细】\n")
        f.write(f"  标签冲突 (相同文本不同标签): {dedup_report['conflict_groups']} 组 → "
                f"删除 {dedup_report['conflict_rows']} 条\n")
        f.write(f"  完全重复 (完全相同文本+标签): 删除 {dedup_report['exact_dup_removed']} 条\n")
        f.write(f"  近重复 (仅URL不同): 删除 {dedup_report['near_dup_removed']} 条\n")
        f.write(f"  训练/验证集重叠: 删除 {overlap_count} 条\n\n")

        f.write("【验证集 — 删除明细】\n")
        f.write(f"  标签冲突: {val_dedup_report['conflict_groups']} 组 → "
                f"删除 {val_dedup_report['conflict_rows']} 条\n")
        f.write(f"  完全重复: 删除 {val_dedup_report['exact_dup_removed']} 条\n")
        f.write(f"  近重复: 删除 {val_dedup_report['near_dup_removed']} 条\n\n")

        f.write("【标签分布变化】\n")
        f.write(f"  训练集 — 清洗前: 0={stats_before_train['label_dist']}  "
                f"清洗后: 0={stats_after_train['label_dist']}\n")
        f.write(f"  验证集 — 清洗前: 0={stats_before_val['label_dist']}  "
                f"清洗后: 0={stats_after_val['label_dist']}\n\n")

        f.write("【清洗步骤】\n")
        f.write("  1. HTML 实体解码 (&amp; → &, &gt; → >, 等)\n")
        f.write("  2. URL 移除 (https?://\\S+ → 空)\n")
        f.write("  3. @mention 规范化 (@\\w+ → @USER，保留结构去个性化)\n")
        f.write("  4. #hashtag 规范化 (#+(\\w+) → \\1，处理 ##双井号)\n")
        f.write("  5. 多余空白规范化 (合并空格、strip)\n")
        f.write("  6. 标签冲突 — 完全相同文本有不同标签 → 全部删除\n")
        f.write("  7. 去重 — 先删完全相同文本+标签，再删仅URL不同的\n")
        f.write("  8. 训练/验证重叠 — 从训练集删除\n\n")

        f.write("【组员使用指南】\n")
        f.write("  直接读取清洗后数据:\n")
        f.write("    train = pd.read_csv('data/train_clean.csv')\n")
        f.write("    val = pd.read_csv('data/val_clean.csv')\n")
        f.write("  推理时清洗新文本:\n")
        f.write("    from src.data_clean import clean_text\n")
        f.write("    text = clean_text(raw_text)  # 与训练数据预处理一致\n")

    print(f"\n清洗报告已保存至: {report_path}")

    # ── 总结 ──
    print(f"\n{'='*60}")
    print(f"  清洗完成!")
    print(f"  训练集: {stats_before_train['rows']} → {stats_after_train['rows']} "
          f"(-{stats_before_train['rows'] - stats_after_train['rows']})")
    print(f"  验证集: {stats_before_val['rows']} → {stats_after_val['rows']} "
          f"(-{stats_before_val['rows'] - stats_after_val['rows']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
