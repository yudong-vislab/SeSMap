# SeSMap Backend Pipeline

This runbook describes the active backend pipeline from raw PDFs to the frontend-ready semantic map data.

All commands run from:

```bash
cd SeSMap-backend
```

## 0. Environment And Models

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
python3 scripts/install_models.py
python3 local_config.py
```

Model assets are declared in `model_requirements.json`.

- BGE encoder: `models/bge-large-en-v1.5/`
- Trained mapper checkpoint: `data/stages/05_model/bert2d_mapper_all_v5.pt`

> Data layout (v2): training-corpus outputs are split by stage under `data/stages/01_corpus … 08_semantic_map`;
> each case is archived under `data/caseN/` (frontend `semantic_map_data.json` + `database-*/summary-*` + `pdf/`);
> misc outputs under `data/outputs/`; old/backup under `data/archive/`. All paths resolve from `local_config.py`.

The mapper checkpoint is private/large. Either copy it to the expected path or set `SESMAP_MAPPER_CKPT_URL` in `.env`.

## 1. One-Command Pipeline

Full run:

```bash
python3 pipeline.py
```

Useful variants:

```bash
# Show commands without running them.
python3 pipeline.py --dry-run

# Skip LLM calls while checking file flow.
python3 pipeline.py --no-llm

# Start from an existing formdatabase.json and existing mapper checkpoint.
python3 pipeline.py --from-stage triplets --skip-train

# Rebuild only frontend semantic map JSON after summaries/case files changed.
python3 pipeline.py --from-stage case_files --to-stage semantic_map
```

## 2. Stage-By-Stage Flow

| Stage | Command | Input | Output |
|---|---|---|---|
| PDF to Markdown | `python3 code_for_data/mineru_pdf.py` | `data/pdf/**/*.pdf` | `data/stages/01_corpus/<paper>/<paper>.md` |
| Markdown to MSUs | `python3 code_for_data/build_corpus.py` | `data/stages/01_corpus/<paper>/<paper>.md` | `data/stages/02_msu/formdatabase.json` (+ `paragraphs.json`, `papers.json`) |
| Raw triplets | `python3 code_for_model/generate_triplets.py` | `data/stages/02_msu/formdatabase.json` | `data/stages/03_triplets/contrastive_triplets_raw.json` |
| Refined triplets | `python3 code_for_model/refine_triplets.py` | raw triplets + formdatabase | `data/stages/03_triplets/contrastive_triplets.json` |
| Embedding cache | `python3 code_for_model/precompute_embeddings.py` | `data/stages/02_msu/formdatabase.json` | `data/stages/04_embeddings/emb_corpus.npy` (+ `.ids.json`) |
| Mapper training | `python3 code_for_model/train_all_v5.py` | refined triplets + formdatabase + emb cache | `data/stages/05_model/bert2d_mapper_all_v5.pt` |
| 2D coordinates | `python3 code_for_data/formdatabase.py` | formdatabase + mapper | `data/stages/02_msu/formdatabase_v2.0.json` |
| Hex/HSU grouping | `python3 code_for_data/generate_hex.py` | `data/stages/02_msu/formdatabase_v2.0.json` | `data/stages/06_hex/hexagon_info.json` |
| HSU summaries | `python3 code_for_data/summarize_hex.py` | hex info + formdatabase_v2.0 | `data/stages/07_summaries/summaries.json` |
| Case files | `python3 code_for_data/build_case_files.py` | formdatabase_v2.0 + summaries | `data/outputs/case_generated/database-*.json`, `summary-*.json` |
| Frontend data | `python3 build_semantic_map_from_db_summary.py --case-dir data/outputs/case_generated --out data/stages/08_semantic_map/semantic_map_data.json` | case files | `data/stages/08_semantic_map/semantic_map_data.json` |

## 3. Standard Data Contracts

### `formdatabase.json`

Flat MSU records:

```json
{
  "idx": 0,
  "MSU_id": 0,
  "sentence": "...",
  "category": "Background|Method|Experiment|Result|Conclusion|Other",
  "rank": 1,
  "type": "text",
  "para_id": 0,
  "paper_id": 0
}
```

### `contrastive_triplets.json`

Training-compatible fields plus inspection metadata:

```json
{
  "anchor": "...",
  "positive": "...",
  "negative": "...",
  "anchor_idx": 0,
  "positive_idx": 1,
  "negative_idx": 2,
  "anchor_context": "...",
  "positive_context": "...",
  "negative_context": "...",
  "source": "embedding_topk"
}
```

### `formdatabase_v2.0.json`

Same as `formdatabase.json`, plus:

```json
{ "2d_coord": [1.23, 4.56] }
```

### `hexagon_info.json` / `summaries.json`

HSU cell records:

```json
{
  "hex_coord": [0, 1],
  "country": 0,
  "MSU_ids": [0, 1, 2],
  "summary": "Only in summaries.json"
}
```

### `semantic_map_data.json`

Frontend-ready data with:

- `subspaces`
- `links`
- `msu_index`
- `indices`
- `stats`

## 4. Active vs Legacy Scripts

Active pipeline scripts live in:

- `code_for_data/mineru_pdf.py`
- `code_for_data/build_corpus.py`
- `code_for_model/generate_triplets.py`
- `code_for_model/refine_triplets.py`
- `code_for_model/precompute_embeddings.py`
- `code_for_model/train_all_v5.py`
- `code_for_data/formdatabase.py`
- `code_for_data/generate_hex.py`
- `code_for_data/summarize_hex.py`
- `code_for_data/build_case_files.py`
- `build_semantic_map_from_db_summary.py`
- `pipeline.py`

Old hard-coded or experimental scripts are archived under `legacy/`.
