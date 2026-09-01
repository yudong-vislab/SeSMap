# old/ — 已归档内容索引

本目录存放**不再支撑当前结论**的文件。移动时间 2026-09-01，全部经 `git mv`，可用
`git log --follow <路径>` 追溯，也可用 `git checkout <commit> -- <原路径>` 还原。

当前有效结论见 `results/` 与 `../SeSMap-backend/code_for_model/eval_*.py`。

---

## 01_superseded_eval/ — 被本轮新评测取代的评测脚本

| 文件 | 被什么取代 | 原因 |
|---|---|---|
| `eval_domain_oos.py` | `eval_bio_transfer.py` | 只测 v10r 坐标、无对照组；新版加了 v11/PCA/UMAP/重拟合五方对照 |
| `eval_oos_v10.py` | `eval_bio_transfer.py` | v10 时期的样本外评测 |
| `eval_contrastive.py` | `eval_merged13.py` | 早期对比学习评测 |
| `eval_faithfulness.py` | `eval_merged13.py` | **含 kNN 差一位 bug**（`kneighbors(X=None)` 后又切 `[:,1:]`），报出的 knnOv 偏低约 0.07 |
| `eval_full.py` | `eval_merged13.py` | 旧 Table 1 生成器，基于 2,689 语料；现主表基于 6,113 合并语料 |
| `eval_seeds.py` | `eval_merged13.py` + `merged13_seeds.json` | 多种子改在合并语料上做，且口径已修正 |
| `eval_case12_projection.py` | `eval_merged13.py` | case1/case2 专用投影评测 |

## 02_dead_pipeline/ — 无人引用或已失效的流水线脚本

| 文件 | 状态 |
|---|---|
| `build_semantic_map.py` | 无任何调用方；线上走 `build_semantic_map_from_db_summary.py` |
| `hex_binning.py` | 无任何调用方；前端 `semanticMap.js` 自带六边形分箱实现 |
| `build_case3.sh` | 被 `build_case3_v11.sh` 取代 |
| `run_bio_eval.sh` | **移动前已失效**：调用 `code_for_model/train_all_v10.py`，该文件早已不在（现位于 `old/04_data_archive/archive/deprecated/code/`） |

## 03_legacy_code/ — 原 `legacy/` 目录

v3/v4 时期的训练与数据脚本，仓库里早已标记为 legacy，原样迁入。

## 04_data_archive/ — 原 `data/archive/` 目录

各 case 在 v7/v8 改版前的快照、已废弃的 checkpoint（v5/v6/v7/v8 系列、
`v11_hybrid_0.3.pt`、`v11_con_only.pt` 等）、以及废弃代码。约 336 MB。

## 05_stale_results/ — 已被重跑取代的结果日志

| 文件 | 取代者 |
|---|---|
| `eval_10k_20260821.log` / `eval_30k_20260827.log` | `m13_eval.log`、`merged13_eval.json` |
| `seeds_20260824.log` | `m13_seeds.log`、`merged13_seeds.json` |
| `oos_fair_20260821.log` | `bio_transfer_20260831.log` |
| `multi_corpus.json` / `multi_corpus_20260818.log` | `merged13_eval.json` |
| `pairs_30k_20260827.log` | 30k 对未用于最终结论；最终用 10k（见保留的 `pairs_10k_20260821.log`） |
| `trainB_20260830.log` | 语料 B 已并入 merged13 统一训练 |

---

## 保留但**数字需复核**的两项

以下留在 `results/`，但其中的 knnOv 可能经由 `models.py::evaluate_quick` 产生，
而该函数在 2026-09-01 之前含 kNN 差一位 bug。若这些数字要进论文，需重跑：

- `subspace_eval_20260821.log`、`subspace_ab_20260830.log`（子空间评测）
- `bz_baselines_20260818.log`、`bz_ours_20260818.log`（边界区评测）

各脚本自身的局部 kNN 实现（显式传 X 再切 `[:,1:]`）是正确的，只有经 `evaluate_quick`
的那一列受影响。`models.py` 已于同日修复，重跑即可得到正确值。
