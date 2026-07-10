# SeSMap

SeSMap 是一个面向科研文献的**语义地图可视分析系统**:把论文拆成细粒度语义单元(MSU),用一个参数化投影把它们映射到 2D,组织成按 discourse role 划分的子空间,并支持跨论文的语义关系检查、LLM 辅助检索/摘要/交互。

系统由两条联动视图构成:

- **语义地图视图**:浏览子空间、HSU 六边形、flight 路线、选中的 MSU;
- **Source Gallery + Stepwise Analysis**:查看论文、保存的路径、勾选的 MSU 与 LLM 摘要。

## 架构

```text
SeSMap/
├── SeSMap-backend/    Flask 后端 + 数据/模型流水线 + RAG + LLM 配置   → 见 SeSMap-backend/README.md
└── SeSMap-frontend/   Vue 3 + Vite 前端(语义地图 UI / Gallery / 分析面板) → 见 SeSMap-frontend/README.md
```

前端通过 Vite 代理把 `/api/*` 转发到后端 `http://127.0.0.1:5000`。

## 端到端流程(概览)

```text
PDF ──MinerU──▶ Markdown ──LLM──▶ MSU 语料 ──bge──▶ 句向量
    ──v10 投影(全批 parametric t-SNE)──▶ 2D 坐标
    ──hex 分箱──▶ HSU ──LLM──▶ 摘要 ──▶ database-*/summary-* ──▶ semantic_map_data.json
    ──▶ 前端渲染:子空间 / boundary zone / flight / Chat / Gallery
```

- **投影模型 = v10**(参数化,直接优化邻域保持):in-sample trust 0.930 / knnOv 0.303,超过非参数 UMAP、逼近 t-SNE,且能把未见句子投进固定坐标系(句子级 OOS 0.268 > `UMAP.transform` 0.222)。
- **设计要点**:论文分离与跨源共址在单一 2D 布局中几何互斥 → **boundary-zone 检测放在高维语义空间,2D 布局专职导航**。
- 详细分阶段命令见后端 README 与 `SeSMap-backend/REPRODUCE.md`。

## 快速开始

**1. 后端**(pyenv 3.10.14;`.python-version` 已固定)

```bash
cd SeSMap-backend
python3 -m pip install -r requirements.txt
cp .env.example .env          # 填 LLM_API_KEY / LLM_BASE_URL
python3 scripts/install_models.py    # 下载 bge 编码器
python3 app.py                # http://127.0.0.1:5000
```

**2. 前端**

```bash
cd SeSMap-frontend
npm install
npm run dev                   # http://localhost:5173
```

启动后默认加载一个 case;在 Chat 里说 "show case1 / case2 / 生信" 切换。

## 从零构建一张地图 / 新增一个 case

把 PDF 放进 `SeSMap-backend/data/pdf/`,按后端 README 第 2 节依次跑
`mineru_pdf → build_corpus → precompute_embeddings → train_all_v10 → formdatabase → generate_hex → summarize_hex → build_case_files → build_semantic_map`;
生信 case3 已封装成一键脚本:`bash SeSMap-backend/build_case3.sh`。

## 主要功能

- 多 case 语义地图(不同研究领域各自建图)。
- 自然语言控制子空间可见性与构造。
- Source Gallery 按主题展示论文缩略图。
- Stepwise Analysis View 保存路线、对勾选 MSU 结构化摘要。
- 基于项目 PDF 的 RAG 问答。
- 集中式 LLM 配置(chat / 意图解析 / RAG / MSU 摘要 / 上下文压缩 / embedding)。
