# models/ 说明

存放分类模型文件，供 `src/evaluate.py` 评估和 `src/inference.py` 推理加载。

| 文件 | 来源脚本 | 是否入库 | 说明 |
|------|----------|----------|------|
| `lr_model.pkl` | `src/train_baseline.py` | ✅ 已提交 | TF-IDF + 逻辑回归，joblib 保存的 `{'model','vectorizer'}` |
| `bigru.pt` | `src/train_bigru.py` | ✅ 已提交 | BiGRU 权重 + 词表 + 结构配置（dict） |
| `bert_model/` | `src/train_bert.py` | ❌ 未入库 | 微调后的 BERT（约 420 MB，被 `.gitignore` 排除）|

## 为什么 bert_model/ 不入库

微调后的 BERT 权重约 420 MB，超出 Git 仓库合理体积，已在 `.gitignore`
中排除。`outputs/` 中已保存它在验证集上的**评估结果（指标表与图）**，
若需复现该模型，本地重新训练即可：

```bash
python src/train_bert.py --epochs 3 --batch_size 16
```

训练完成后会自动生成 `models/bert_model/`，再运行
`python src/evaluate.py` 即可把 BERT 纳入三模型对比。

> 注：`evaluate.py` 在缺少某个模型文件时会自动跳过该模型，
> 因此仅克隆仓库（无 `bert_model/`）也能直接评估 LR 与 BiGRU。
