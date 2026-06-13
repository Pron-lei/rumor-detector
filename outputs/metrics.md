# 验证集评估结果

验证集：`data\val_clean.csv`，共 400 条。

| model | accuracy | precision_macro | recall_macro | f1_macro | precision_rumor | recall_rumor | f1_rumor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + LR | 0.8675 | 0.8753 | 0.857 | 0.8624 | 0.906 | 0.7759 | 0.8359 |
| BiGRU | 0.8275 | 0.8337 | 0.8156 | 0.8205 | 0.8571 | 0.7241 | 0.785 |
| BERT | 0.845 | 0.8423 | 0.8423 | 0.8423 | 0.8218 | 0.8218 | 0.8218 |

- `*_macro`：两类的宏平均；`*_rumor`：谣言类（标签=1）单独指标。
