"""
TF-IDF + 逻辑回归 基线模型训练
- 读取清洗后的 train_clean.csv 训练
- 在 val_clean.csv 上快速自评
- 保存 models/lr_model.pkl，格式为 {'model': ..., 'vectorizer': ...}

注：这是为了让 D 的评估流程能跑通而提供的基线实现，
    B 可在此基础上继续调参或替换为更强的版本，只要保持
    保存格式（joblib dump 一个含 model / vectorizer 的 dict）即可。
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def preprocess(series: pd.Series) -> pd.Series:
    """与推理阶段一致的小写 + 去标点。"""
    return series.str.lower().str.replace(r"[^\w\s]", "", regex=True)


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train_clean.csv"))
    val_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "val_clean.csv"))

    X_train = preprocess(train_df["text"].astype(str))
    y_train = train_df["label"]
    X_val = preprocess(val_df["text"].astype(str))
    y_val = val_df["label"]

    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 3),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=3000, C=2.0, solver="liblinear", random_state=42)
    model.fit(X_train_vec, y_train)

    val_pred = model.predict(X_val_vec)
    print(f"Val Acc: {accuracy_score(y_val, val_pred):.4f}")
    print(classification_report(y_val, val_pred, digits=4))

    out_path = os.path.join(PROJECT_ROOT, "models", "lr_model.pkl")
    joblib.dump({"model": model, "vectorizer": vectorizer}, out_path)
    print(f"模型已保存为 {out_path}")


if __name__ == "__main__":
    main()
