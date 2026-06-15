"""
端到端推理 Pipeline + RumourDetectClass
- 输入一条推文文本
- 输出分类结果 + 置信度 + 中文判断依据（LLM 模块到齐后启用）

设计要点：
  - torch / transformers 采用延迟导入，仅在选择对应模型时才加载；
    因此即使只安装了 sklearn + joblib，也能正常使用 LR 模型推理。
  - 三种模型（LR / BiGRU / BERT）共享统一的 classify() 入口。
  - 解释模块缺失时自动降级为纯分类模式。
"""
import os
import sys
import re
import argparse
import numpy as np
import joblib

from data_clean import clean_text

# ── 尝试导入解释模块（成员 C 提交前不可用，只做分类） ──
try:
    from explain import generate_explanation
    _HAS_EXPLAIN = True
except ImportError:
    _HAS_EXPLAIN = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════
# BiGRU 模型定义（与 train_bigru.py 结构一致，供推理与评估共用）
# ═══════════════════════════════════════════════════════════════════

class BiGRUInference:
    """BiGRU 推理模型，结构必须与 train_bigru.py 保持同步。

    采用工厂函数模式：调用 ``BiGRUInference.from_checkpoint(path)``
    自动从 checkpoint 中推断结构参数并加载权重。
    """

    @staticmethod
    def _build_model(vocab_size, emb_dim, hidden_dim):
        import torch
        import torch.nn as nn

        class _BiGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                self.bigru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(hidden_dim * 2, 1)

            def forward(self, x):
                emb = self.embedding(x)
                _, h = self.bigru(emb)
                h = torch.cat([h[0], h[1]], dim=1)
                return self.fc(h).squeeze(1)

        return _BiGRU()

    @staticmethod
    def from_checkpoint(ckpt_path, device="cpu"):
        """从 checkpoint 加载 BiGRU 模型。

        Returns:
            (model, vocab, max_len)
        """
        import torch

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise ValueError(
                "BiGRU checkpoint 格式不兼容：期望 dict 包含 'model_state' 键。"
            )

        state = ckpt["model_state"]
        vocab = ckpt.get("vocab")
        cfg = ckpt.get("config", {})

        # 从权重形状反推结构，避免与训练超参不一致
        emb_dim = cfg.get("embedding_dim", state["embedding.weight"].shape[1])
        hidden_dim = cfg.get("hidden_dim", state["bigru.weight_hh_l0"].shape[1])
        vocab_size = state["embedding.weight"].shape[0]
        max_len = cfg.get("max_len", 64)

        model = BiGRUInference._build_model(vocab_size, emb_dim, hidden_dim)
        model.load_state_dict(state)
        model.to(device).eval()

        return model, vocab, max_len


# ═══════════════════════════════════════════════════════════════════
# RumourDetectClass — 统一推理入口
# ═══════════════════════════════════════════════════════════════════

class RumourDetectClass:
    """
    谣言检测推理类

    用法:
        detector = RumourDetectClass()          # 默认 BERT（需要 torch + transformers）
        detector = RumourDetectClass("bigru")   # BiGRU（需要 torch）
        detector = RumourDetectClass("lr")      # 逻辑回归（仅需 sklearn + joblib）

        result = detector.classify("推文内容")
        # result: {"label": 0|1, "label_text": "非谣言"|"谣言",
        #          "confidence": 0.87, "explanation": "...", "model": "bert"}
    """

    def __init__(self, model_type: str = "bert"):
        if model_type not in ("bert", "bigru", "lr"):
            raise ValueError(
                f"不支持的模型类型: {model_type}，可选 bert / bigru / lr"
            )

        self.model_type = model_type
        self.model = None
        self.tokenizer = None     # BERT
        self.vocab = None         # BiGRU
        self._bigru_max_len = 64  # BiGRU
        self.vectorizer = None    # LR

        self._load_model()

    # ── 模型加载（延迟导入重型依赖） ──

    def _load_model(self):
        loaders = {
            "bert": self._load_bert,
            "bigru": self._load_bigru,
            "lr": self._load_lr,
        }
        loaders[self.model_type]()

    def _load_bert(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "BERT 模型需要 transformers 库。请运行: pip install transformers"
            )
        import torch

        model_dir = os.path.join(PROJECT_ROOT, "models", "bert_model")
        if not os.path.isdir(model_dir) or not os.listdir(model_dir):
            raise FileNotFoundError(
                f"BERT 模型未找到: {model_dir}\n"
                "  请先运行 python src/train_bert.py 训练，或按 models/README.md 下载。"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device).eval()
        print(f"[RumourDetect] BERT 模型已加载 (device={self._device})")

    def _load_bigru(self):
        import torch

        model_path = os.path.join(PROJECT_ROOT, "models", "bigru.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BiGRU 模型未找到: {model_path}")

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, self.vocab, self._bigru_max_len = BiGRUInference.from_checkpoint(
            model_path, device=self._device
        )
        self.model = model
        print(f"[RumourDetect] BiGRU 模型已加载 (device={self._device})")

    def _load_lr(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "lr_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LR 模型未找到: {model_path}")
        data = joblib.load(model_path)
        self.model = data["model"]
        self.vectorizer = data["vectorizer"]
        print("[RumourDetect] LR 模型已加载")

    # ── 预处理 ──

    def _preprocess(self, text: str) -> str:
        """与训练一致的预处理：clean_text → 小写 → 去标点"""
        text = clean_text(text)
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    # ── 分类预测 ──

    def _predict_bert(self, text: str) -> "tuple[int, float]":
        import torch

        encoding = self.tokenizer(
            text,
            max_length=64,          # 与 train_bert.py --max_len 默认值一致
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        return pred, float(probs[pred])

    def _predict_bigru(self, text: str) -> "tuple[int, float]":
        import torch
        from text_utils import encode

        cleaned = self._preprocess(text)
        ids = torch.tensor(
            encode(cleaned, self.vocab, self._bigru_max_len), dtype=torch.long
        ).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logit = self.model(ids)
            prob = torch.sigmoid(logit).item()
        if prob > 0.5:
            return 1, prob
        return 0, 1.0 - prob

    def _predict_lr(self, text: str) -> "tuple[int, float]":
        cleaned = self._preprocess(text)
        X = self.vectorizer.transform([cleaned])
        pred = int(self.model.predict(X)[0])
        # 尝试获取概率；若 sklearn 版本不兼容则退化为 1.0
        try:
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba[pred])
        except Exception:
            confidence = 1.0
        return pred, confidence

    # ── 统一入口 ──

    def classify(self, text: str, with_explanation: bool = True) -> dict:
        """
        Args:
            text: 推文文本
            with_explanation: 是否生成 LLM 解释

        Returns:
            {"label": 0|1, "label_text": "非谣言"|"谣言",
             "confidence": 0.87, "explanation": "...", "model": "bert"}
        """
        predictors = {
            "bert": self._predict_bert,
            "bigru": self._predict_bigru,
            "lr": self._predict_lr,
        }
        label, confidence = predictors[self.model_type](text)

        label_text = "谣言" if label == 1 else "非谣言"
        explanation = ""

        if with_explanation:
            if _HAS_EXPLAIN:
                try:
                    explanation = generate_explanation(text, label, confidence)
                except Exception as e:
                    explanation = f"[解释生成失败: {e}]"
            else:
                explanation = (
                    "[解释模块未就绪] "
                    "LLM 解释模块（explain.py / llm_api.py）尚未提交，"
                    "请等待成员 C 合并后重试。当前仅输出分类结果。"
                )

        return {
            "label": label,
            "label_text": label_text,
            "confidence": confidence,
            "explanation": explanation,
            "model": self.model_type,
        }


# ═══════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════

def _print_result(result: dict, text: str):
    label_emoji = "🟢" if result["label"] == 0 else "🔴"
    print(f"\n{'=' * 60}")
    print(f"  {label_emoji} 谣言检测结果")
    print(f"{'=' * 60}")
    print(f"  推文: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"  模型: {result['model']}")
    print(f"  标签: {result['label_text']} ({result['label']})")
    print(f"  置信度: {result['confidence']:.2%}")
    if result.get("explanation"):
        print(f"  判断依据: {result['explanation']}")
    print(f"{'=' * 60}")


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="可解释的谣言检测")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--text", "-t", type=str, default=None, help="单条检测")
    mode.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument(
        "--model", "-m", type=str, default="bert",
        choices=["bert", "bigru", "lr"],
        help="分类模型（默认 bert）",
    )
    parser.add_argument("--no-explain", action="store_true", help="跳过解释")
    args = parser.parse_args()

    print(f"正在加载 {args.model.upper()} 模型...", flush=True)
    try:
        detector = RumourDetectClass(model_type=args.model)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        return
    except ImportError as e:
        print(f"\n[错误] 缺少依赖: {e}")
        return

    with_explanation = not args.no_explain
    if not _HAS_EXPLAIN and with_explanation:
        print("[提示] 解释模块未就绪，仅输出分类结果。")

    # 单条模式
    if args.text:
        result = detector.classify(args.text, with_explanation=with_explanation)
        _print_result(result, args.text)
        return

    # 交互模式（默认）
    print(f"\n{'=' * 60}")
    print(f"  交互模式（模型: {args.model.upper()}）")
    print(f"  输入推文回车检测，输入 q 退出")
    print(f"{'=' * 60}")

    while True:
        try:
            text = input("\n推文> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if text.lower() in ("q", "退出"):
            print("退出。")
            break
        if not text:
            continue
        result = detector.classify(text, with_explanation=with_explanation)
        _print_result(result, text)


if __name__ == "__main__":
    main()
