# SeSMap

SeSMap is a **semantic-map visual analytics system** for scientific literature. It decomposes papers into fine-grained Minimal Semantic Units (MSUs), projects them into 2D with a parametric model, organizes them into discourse-role subspaces, and supports cross-paper semantic inspection with LLM-assisted retrieval, summarization, and interaction.

## Architecture

```text
SeSMap/
├── SeSMap-backend/    Flask backend + data/model pipeline + RAG   → SeSMap-backend/README.md
└── SeSMap-frontend/   Vue 3 + Vite frontend (map UI / gallery / chat) → SeSMap-frontend/README.md
```

The frontend proxies `/api/*` to the backend at `http://127.0.0.1:5000`.

## Pipeline Overview

```text
PDF → Markdown (MinerU) → MSU corpus (LLM) → sentence embeddings (bge)
    → 2D coordinates (v10 parametric projection) → hex binning (HSU)
    → HSU summaries → case files → semantic_map_data.json → frontend rendering
```

## Quick Start

**1. Backend** (Python 3.10)

```bash
cd SeSMap-backend
python3 -m pip install -r requirements.txt
cp .env.example .env                 # set LLM_API_KEY / LLM_BASE_URL
python3 scripts/install_models.py    # download the bge encoder
python3 app.py                       # http://127.0.0.1:5000
```

**2. Frontend**

```bash
cd SeSMap-frontend
npm install
npm run dev                          # http://localhost:5173
```

## Building a New Case

Put PDFs into `SeSMap-backend/data/pdf/` and run the pipeline steps in order (see the backend README):

```text
mineru_pdf → build_corpus → precompute_embeddings → train_all_v10
→ formdatabase → generate_hex → summarize_hex → build_case_files
→ build_semantic_map → extract_thumbnails
```

## Main Features

- Multiple cases: each corpus builds its own semantic map.
- Natural-language control of subspace visibility and construction.
- Source Gallery with automatic paper thumbnails; selecting papers filters the map dynamically.
- Stepwise Analysis View for saved routes and structured MSU summaries.
- RAG question answering over the project PDFs.
