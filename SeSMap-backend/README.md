# SeSMap Backend

Flask 后端 + 数据/模型流水线:把科研 PDF 处理成 MSU 语义单元,训练参数化投影得到 2D 语义地图,切分成 discourse-role 子空间,并为前端提供 semantic-map / 子空间构造 / RAG 问答 / LLM 配置等 API。

> 详细的分阶段输入输出表见 [`REPRODUCE.md`](REPRODUCE.md)。本文件是总览 + 主要命令。

---

## 1. 环境准备(一次性)

| 组件 | 说明 |
|---|---|
| Python | **pyenv 3.10.14**(`.python-version` 已固定,进目录自动切换) |
| 主环境依赖 | `python3 -m pip install -r requirements.txt`(torch / sentence-transformers / sklearn / umap / flask / langchain / faiss …) |
| BGE 编码器 | `models/bge-large-en-v1.5/`(1024 维);`python3 scripts/install_models.py` 或 huggingface 下载 |
| MinerU(PDF→MD) | 独立 venv,和主环境隔离:`/opt/homebrew/bin/python3.12 -m venv .venv-mineru && .venv-mineru/bin/pip install -U "mineru[core]"`。`mineru_pdf.py` 会自动调用它,无需 activate |
| LLM 配置 | `cp .env.example .env`,填 `LLM_API_KEY` / `LLM_BASE_URL`(走 OpenAI 兼容端点)/ 模型名 |

路径中枢是 [`local_config.py`](local_config.py)——所有脚本从这里取路径,可用 `.env` 里的同名环境变量覆盖:

```bash
python3 local_config.py   # 打印解析后的所有路径,确认 bge_exists=True
```

数据归档结构:
```
data/
├── pdf/                    输入 PDF
├── stages/01_corpus … 08_semantic_map   训练主语料的分阶段产物
├── case1/ case2/ case3/   每个 case 的前端数据(semantic_map_data.json + database-*/summary-* + pdf/)
├── indexes/caseN/         RAG 的 FAISS 索引
├── outputs/               评测结果/缓存
└── archive/               旧文件/备份
```

---

## 2. 从 PDF 构建一张语义地图(完整流程)

把 PDF 放进 `data/pdf/`,依次执行(隔离到独立语料时,给每条命令加 `SESMAP_STAGES_DIR=data/<name>/stages` 前缀):

```bash
# ① PDF -> Markdown(MinerU;首次自动下模型)
python3 code_for_data/mineru_pdf.py

# ② Markdown -> MSU 语料(LLM 抽取,需 .env 的 key)
python3 code_for_data/build_corpus.py            # 仅解析、不调 LLM: 加 --no-llm

# ③ 句向量缓存(bge 冻结,只算一次;训练/评测都读它)
python3 code_for_model/precompute_embeddings.py

# ④ 训练投影模型 v10(全批 parametric t-SNE,直接优化邻域保持)
python3 code_for_model/train_all_v10.py

# ⑤ 用 v10 给每个 MSU 生成 2D 坐标
python3 code_for_data/formdatabase.py

# ⑥ HSU 六边形分箱
python3 code_for_data/generate_hex.py

# ⑦ HSU 摘要(LLM;--no-llm 用前 3 句兜底)
python3 code_for_data/summarize_hex.py

# ⑧ 切分成 database-*/summary-* 并合成前端数据
python3 code_for_data/build_case_files.py --out-dir data/caseN
python3 build_semantic_map_from_db_summary.py --case-dir data/caseN --out data/caseN/semantic_map_data.json

# ⑨ gallery 缩略图 + manifest(自动为每篇论文选主图 = Figure 1 优先)
python3 code_for_data/extract_thumbnails.py --corpus <corpus>/01_corpus \
    --papers <corpus>/02_msu/papers.json --case caseN --source-offset <N>
```

> **Gallery(方案 B,泛化)**:`extract_thumbnails.py` 从 MinerU 产物为每篇论文选一张主图(caption 以 Figure 1 开头优先),缩略后存 `data/caseN/thumbnails/`,并写 `data/caseN/gallery.json`(paper→缩略图→地图国家/来源映射)。后端 `GET /api/gallery?project_id=caseN` + `GET /api/gallery/thumb/<case>/<file>` 提供数据;前端读 manifest 渲染,**未来上传任意论文跑完流程即自动接入 gallery,无需改前端代码**。

> **模型说明**:最终投影模型是 **v10**(`train_all_v10.py`,全批 parametric t-SNE),`local_config.MAPPER_CKPT` 默认指向它。`train_all_v5.py`(triplet)、`generate_triplets.py`/`refine_triplets.py` 是早期迭代,v10 直接用 ④ 的句向量缓存、不需要三元组。
>
> **坐标尺度**:v10 输出尺度较大,作为独立 case 接入前端时需缩放到与 case1/2 一致(见 `build_case3.sh`),否则 hex 网格过大、前端只显示几格。

---

## 3. 评测

```bash
python3 code_for_model/eval_faithfulness.py     # in-sample: Ours vs PCA/tSNE/UMAP(trust/knnOv@12)
python3 code_for_model/eval_oos_v10.py          # 零泄漏 OOS: Ours(v10) vs UMAP.transform(句子级泛化)
python3 code_for_model/eval_domain_oos.py --domain-formdb ... --domain-cache ...   # 跨领域迁移对照
python3 code_for_model/detect_boundary_pairs.py # boundary-zone 高维检测 -> data/outputs/bz_pairs.json
```

当前语料(服务器恢复)= 2689 MSU / 6 篇。v10 in-sample:trust **0.930** / knnOv **0.303**(超 UMAP,平 t-SNE);句子级 OOS 0.268 > UMAP.transform 0.222。

---

## 4. Cases(前端数据)

前端每个 case 读 `data/<caseId>/semantic_map_data.json`。已内置 case1(scramjet 燃烧)、case2(时空可视化/大气)、case3(基因组可视化)。

```bash
bash build_case3.sh     # 用 data/bio_eval 的生信语料一键构建 case3(含坐标缩放,对齐 case1/2)
```

---

## 5. 启动服务

```bash
python3 app.py          # Flask,默认 127.0.0.1:5000(FLASK_HOST/FLASK_PORT 可改)
```

主要 API:`GET /api/semantic-map?project_id=case1` · `POST /api/subspaces` · `POST /api/rag/index` · `POST /api/query` · `GET /api/llm/config`。前端通过 Vite 代理连到这里(见前端 README)。
