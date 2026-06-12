# 验证集评估结果

验证集：`data\val_clean.csv`，共 400 条。

| model | accuracy | precision_macro | recall_macro | f1_macro | precision_rumor | recall_rumor | f1_rumor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + LR | 0.8175 | 0.8158 | 0.8114 | 0.8132 | 0.8061 | 0.7644 | 0.7847 |
| BiGRU | 0.8275 | 0.8337 | 0.8156 | 0.8205 | 0.8571 | 0.7241 | 0.785 |
| BERT | 0.8325 | 0.8304 | 0.828 | 0.829 | 0.8166 | 0.7931 | 0.8047 |

- `*_macro`：两类的宏平均；`*_rumor`：谣言类（标签=1）单独指标。
