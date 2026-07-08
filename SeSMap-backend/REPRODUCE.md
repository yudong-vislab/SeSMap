# SeSMap 复现流程 Runbook

> 目标：从原始 PDF 从 0 复现整篇论文的数据→模型→可视化→评测流程（本地、可复现）。
> 主体思路不变，只做优化：投影模型改 **v5（语义优先，paper_id 只上色）**，评测改**泛化/OOS + 构造数据**。
>
> **状态图例**：✅ 已本地化并验证 ｜ 🟢 已写好可直接用 ｜ ⚠️ 待本地化（仍有 `/home/lxy` 或 `pollution_result/` 服务器路径，下一步修）
>
> 所有命令默认在后端根目录执行：
> ```bash
> cd /Users/yudong/Desktop/SeSMap/SeSMap-backend
> ```
> 除 MinerU 外全部用**主环境 python3**（3.9，已装 torch/sentence-transformers/openai/dotenv/sklearn/umap）。
> MinerU 单独在 `.venv-mineru`（python3.12），由 `mineru_pdf.py` 自动调用，**无需手动 activate**。

---

## 0. 环境准备（一次性）

```bash
# 0.1 主环境依赖（若缺）
python3 -m pip install -r requirements.txt

# 0.2 安装模型资产（不会提交到 GitHub）
#     - BGE 编码器下载到 models/bge-large-en-v1.5/
#     - mapper checkpoint 放到 data/outputs/bert2d_mapper_all_v5.pt
#       私有下载地址可写入 .env: SESMAP_MAPPER_CKPT_URL=...
python3 scripts/install_models.py

# 0.3 MinerU 隔离环境（PDF→MD 专用，和主环境完全隔离）
/opt/homebrew/bin/python3.12 -m venv .venv-mineru
.venv-mineru/bin/python -m pip install -U pip
.venv-mineru/bin/pip install -U "mineru[core]"
.venv-mineru/bin/mineru --version   # 应为 3.x

# 0.4 .env（LLM 走 dmxapi 代理；本地路径）
#   必填：LLM_API_KEY / LLM_BASE_URL=https://ssvip.dmxapi.com/v1 / LLM_CHAT_MODEL=gpt-4o
#   已加：BGE_MODEL_PATH / SESMAP_DATA_ROOT（见 local_config.py）
```

**路径中枢**：`local_config.py`（所有脚本从这里取本地路径，可用同名环境变量覆盖）
```bash
python3 local_config.py          # 打印解析后的路径，确认 bge_exists=True   ✅
```
目录约定：`data/pdf/`（输入 PDF）→ `data/corpus/<name>/`（中间）→ `data/outputs/`（产物）

---

## 1. 数据侧：PDF → MSU 库

| 步 | 命令 | 输入 → 输出 | 状态 |
|---|---|---|---|
| ① PDF→MD | `python3 code_for_data/mineru_pdf.py` | `data/pdf/**.pdf` → `data/corpus/<name>/<name>.md` | ✅ |
| ②③④ MD→MSU库 | `python3 code_for_data/build_corpus.py` | `corpus/<name>/<name>.md` → `data/outputs/formdatabase.json` | ✅ 已跑通 |

```bash
# ① 把待复现 PDF 放进 data/pdf/（可放子目录），然后：
python3 code_for_data/mineru_pdf.py
#   首次会自动下载几 GB 的 MinerU pipeline 模型（一次性）；后端锁定 pipeline（CPU 友好）

# ②③④ 解析分节 + LLM 抽 MSU + 汇总（③调 gpt-4o，需 .env 里的 key）
python3 code_for_data/build_corpus.py
#   仅想先跑通管线、不调 LLM：
python3 code_for_data/build_corpus.py --no-llm
```
产物 `formdatabase.json` 每条：`{idx, MSU_id, sentence, category(role), rank, type, para_id, paper_id}`
（图 MSU 可选：`get_figureMSU.py`，多模态 LLM，⚠️待本地化）

---

## 2. 模型侧：三元组 → 缓存 → 训练 v5 → 加坐标

| 步 | 命令 | 输入 → 输出 | 状态 |
|---|---|---|---|
| ⑥ 三元组 | `python3 code_for_model/generate_tri_withinfo_v3.py` | `formdatabase.json` → `contrastive_triplets.json` | ⚠️ 待本地化 |
| 缓存 | `python3 code_for_model/precompute_embeddings.py --corpus data/outputs/formdatabase.json --out data/outputs/emb_corpus.npy` | 句子 → `emb_corpus.npy` | 🟢 |
| ⑦ 训练 v5 | `python3 code_for_model/train_all_v5.py` | triplets + emb → `bert2d_mapper_all_v5.pt` | 🟢（`__main__` 路径需改本地/缓存） |
| ⑤/⑧ 加坐标 | `python3 code_for_data/formdatabase.py` | `formdatabase.json` + `v5.pt` → `formdatabase_v2.0.json`（+2d_coord） | ⚠️ 待本地化 + 指向 v5 |

> v5 相对 v3 的改动（`train_all_v5.py`）：删掉把不同论文推远(target=30)/拉近(=10)的身份分支，
> 只留同段落弱就近 + 语义 triplet + repulsion + stabilization(+可选 role)。**paper_id 只上色，不进损失。**
> 消融：把 `lambda_para / lambda_st / lambda_role` 逐个置零重训对比。

---

## 3. 可视化数据：HSU 六边形 + 摘要

| 步 | 命令 | 输入 → 输出 | 状态 |
|---|---|---|---|
| ⑨ Hex/HSU | `python3 code_for_data/generate_hex.py` | `formdatabase_v2.0.json` → `hexagon_info.json` | ⚠️ 待本地化 |
| ⑩ HSU 摘要 | `python3 code_for_data/summarize_hex.py` | hex + db → `summaries.json`（LLM） | ⚠️ 待本地化 |

---

## 4. 评测

| 步 | 命令 | 说明 | 状态 |
|---|---|---|---|
| in-sample 诊断 | `python3 code_for_model/eval_layout.py` | 当前布局 vs PCA/t-SNE/UMAP：语义保真 / role / paper 三组指标 | 🟢 已跑过 |
| 泛化/OOS | （释义集生成 + OOS 评测脚本） | 未见句子的保真、释义共位、跨域、BZ 构造 | ⬜ 待写 |

> 说明：`eval_layout.py` 是"诊断/原型"（case 论文少、可能在训练集内），非最终可发表数字；
> 最终数字应在 held-out 语料 + 构造集上，用泛化口径出。

---

## 5. 运行时（可视分析系统）

```bash
python3 app.py            # Flask，读 .env；前端另起   🟢 结构完好
```

---

## 待办（我下一步做的本地化）
⚠️ 标记的脚本仍有服务器硬编码路径，需要改成读 `local_config.py`：
`generate_tri_withinfo_v3.py`、`formdatabase.py`(加坐标)、`generate_hex.py`、`summarize_hex.py`、`get_figureMSU.py`、`inference_interactive_v2.py`。
改完会各自加一个 smoke test（合成数据/现有 case 数据跑通），再更新本表状态。
