# legacy/ — 已归档、被取代的脚本

这些脚本已被新的本地流水线取代，移到这里存档（git 有历史，随时可恢复；确认不用了可整目录删）。

| 归档文件 | 被谁取代 | 原因 |
|---|---|---|
| `marker_pdf.py` | `code_for_data/mineru_pdf.py` | PDF→MD 改用 MinerU |
| `generate_dict.py` | `code_for_data/build_corpus.py`（`aggregate`） | MSU 汇总已并入 build_corpus |
| `train_all_v4.py` | `code_for_model/train_all_v5.py` | v4 是加了 paper 分离损失(paper_center/diff)的实验分支，未用于论文，被 v5 取代 |

恢复某个文件：`git mv legacy/<file> <原路径>` 或 `mv legacy/<file> ../<原路径>`。
