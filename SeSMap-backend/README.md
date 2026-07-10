# SeSMap Backend

Flask backend + data/model pipeline: turns scientific PDFs into MSU semantic units, trains a parametric 2D projection, builds discourse-role subspaces, and serves the semantic-map / subspace / RAG APIs for the frontend.

## 1. Setup (one-time)

```bash
python3 -m pip install -r requirements.txt          # Python 3.10 (pyenv, pinned in .python-version)
cp .env.example .env                                # set LLM_API_KEY / LLM_BASE_URL
python3 scripts/install_models.py                   # bge-large-en-v1.5 encoder → models/
# MinerU (PDF → Markdown) lives in its own venv:
/opt/homebrew/bin/python3.12 -m venv .venv-mineru && .venv-mineru/bin/pip install -U "mineru[core]"
```

All paths are centralized in `local_config.py` (overridable via env vars). Data layout:

```text
data/
├── pdf/            input PDFs
├── stages/         staged pipeline outputs (01_corpus … 08_semantic_map)
├── case1..3/       per-case frontend data
└── outputs/        evaluation results
```

## 2. Pipeline: PDF → Semantic Map

Put PDFs into `data/pdf/`, then run in order:

```bash
python3 code_for_data/mineru_pdf.py               # ① PDF → Markdown
python3 code_for_data/build_corpus.py             # ② Markdown → MSU corpus (LLM)
python3 code_for_model/precompute_embeddings.py   # ③ sentence embeddings (bge)
python3 code_for_model/train_all_v10.py           # ④ train the v10 projection
python3 code_for_data/formdatabase.py             # ⑤ 2D coordinates for every MSU
python3 code_for_data/generate_hex.py             # ⑥ hex binning → HSUs
python3 code_for_data/summarize_hex.py            # ⑦ HSU summaries (LLM)
python3 code_for_data/build_case_files.py --out-dir data/caseN     # ⑧ case files
python3 build_semantic_map_from_db_summary.py --case-dir data/caseN --out data/caseN/semantic_map_data.json
python3 code_for_data/extract_thumbnails.py --case caseN ...       # ⑨ gallery thumbnails
```

`bash build_case3.sh` runs the whole case3 build in one command.

## 3. Evaluation

```bash
python3 code_for_model/eval_faithfulness.py       # in-sample vs PCA / t-SNE / UMAP
python3 code_for_model/eval_oos_v10.py            # leak-free out-of-sample
python3 code_for_model/detect_boundary_pairs.py   # boundary-zone detection (high-D)
```

## 4. Run the Server

```bash
python3 app.py        # Flask, http://127.0.0.1:5000
```

Main APIs: `GET /api/semantic-map` · `POST /api/subspaces` · `POST /api/query` · `GET /api/gallery`.
