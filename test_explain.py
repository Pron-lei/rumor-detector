"""解释效果抽查 — 从验证集随机采样，展示分类 + LLM 解释结果。

用法:
    python test_explain.py              # 默认抽查 5 条（不使用 LLM，快速验证模板）
    python test_explain.py -n 10        # 抽查 10 条
    python test_explain.py -n 3 --llm   # 指定数量 + 启用 LLM（需配置 .env）
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.explain import generate_explanation  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VAL_PATH = os.path.join(PROJECT_ROOT, "data", "val_clean.csv")


def main():
    parser = argparse.ArgumentParser(description="解释效果抽查")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=5,
        help="抽查样本数（默认 5）",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="启用 LLM 生成解释（默认使用规则模板）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）",
    )
    args = parser.parse_args()

    if not os.path.exists(VAL_PATH):
        print(f"验证集不存在: {VAL_PATH}")
        print("请先运行 python src/data_clean.py")
        return

    df = pd.read_csv(VAL_PATH)
    n = min(args.num_samples, len(df))
    samples = df.sample(n=n, random_state=args.seed)

    print(f"从验证集随机抽取 {n} 条样本（共 {len(df)} 条），LLM={'开' if args.llm else '关'}\n")

    for i, (_, row) in enumerate(samples.iterrows(), 1):
        text = str(row["text"])
        label = int(row["label"])
        confidence = 0.85  # 抽查阶段使用固定置信度

        explanation = generate_explanation(
            text, label, confidence, use_llm=args.llm,
        )

        print(f"{'─' * 60}")
        print(f"[{i}/{n}]  真实标签: {'谣言' if label == 1 else '非谣言'} ({label})")
        print(f"  推文: {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"  解释: {explanation}")
        print()


if __name__ == "__main__":
    main()
