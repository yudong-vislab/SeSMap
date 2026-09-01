# legacy/ — 已归档、被取代的脚本

这些脚本已被新的本地流水线取代，移到这里存档（git 有历史，随时可恢复；确认不用了可整目录删）。

| 归档文件 | 被谁取代 | 原因 |
|---|---|---|
| `marker_pdf.py` | `code_for_data/mineru_pdf.py` | PDF→MD 改用 MinerU |
| `generate_dict.py` | `code_for_data/build_corpus.py`（`aggregate`） | MSU 汇总已并入 build_corpus |
| `train_all_v4.py` | `code_for_model/train_all_v5.py` | v4 是加了 paper 分离损失(paper_center/diff)的实验分支，未用于论文，被 v5 取代 |
| `code_for_model/generate_tri_withinfo_v3.py` | `code_for_model/generate_triplets.py` + `refine_triplets.py` | 旧脚本硬编码 `/home/lxy/bgemodel`，且直接加载成功时 `embed_model` 未赋值 |
| `code_for_model/train_all_v3.py` | `code_for_model/train_all_v5.py` | v3 使用旧 paper 分离逻辑，主流程改为 v5 |
| `code_for_model/inference_interactive_v2.py` | `code_for_data/formdatabase.py` | 旧交互 demo 硬编码模型路径，主流程只保留批处理坐标生成 |
| `code_for_data/formdatabase2.py` | `code_for_data/formdatabase.py` | 旧格式转换脚本与当前标准数据格式重复 |
| `code_for_data/new_link.py` | 前端动态 link/selection 逻辑 | 实验性连线生成器，路径和输出格式不属于当前 PDF→semantic_map 流程 |
| `code_for_data/figure_tack.py` | 暂无活跃替代 | 旧 figure 补丁脚本，硬编码服务器路径，不属于当前文本 MSU 主流程 |
| `code_for_data/get_figureMSU.py` | 暂无活跃替代 | 旧图像 MSU 抽取试验，硬编码 `case_engine` 路径 |

恢复某个文件：`git mv legacy/<file> <原路径>` 或 `mv legacy/<file> ../<原路径>`。
