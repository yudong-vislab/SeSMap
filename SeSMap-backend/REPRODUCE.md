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
- Active trained mapper checkpoint: `data/stages/05_model/bert2d_mapper_all_v7_pure.pt`
- Legacy mapper checkpoints: `data/stages/05_model/bert2d_mapper_all_v5.pt`, `data/stages/05_model/bert2d_mapper_all_v6.pt`

> Data layout (v2): training-corpus outputs are split by stage under `data/stages/01_corpus … 08_semantic_map`;
> each case is archived under `data/caseN/` (frontend `semantic_map_data.json` + `database-*/summary-*` + `pdf/`);
> misc outputs under `data/outputs/`; old/backup under `data/archive/`. All paths resolve from `local_config.py`.

The mapper checkpoint is private/large. Either copy it to the expected path or set `SESMAP_MAPPER_CKPT_URL` in `.env`.

## 1. Current Mapper Logic: v7

The current semantic projection model is v7, implemented in:

```text
code_for_model/train_all_v7.py
```

v7 replaces the older sparse-triplet mapper with a topology-distilled semantic projection:

- Input: frozen BGE-large sentence embedding, 1024 dimensions.
- Mapper: `ResidualProjectionMapper`, not the old three-layer MLP.
- Architecture:
  - `Linear(1024 -> width)`
  - `LayerNorm + GELU`
  - residual feed-forward blocks with `LayerNorm`, `Linear`, `GELU`
  - 2D projection head
- Default size: `width=512`, `num_blocks=4`, `dropout=0.0`.
- No `BatchNorm`.
- No batch-wise output normalization such as `out / out.max * 10`.
- Training target: distill a corpus-level UMAP semantic layout from the BGE embedding space.
- Optional global KNN triplet and repulsion terms remain in the script, but default to `0.0` because pure target distillation best matches the current faithfulness criterion.
- `paper_id` is not used as a training signal; it remains a display/coloring attribute.

Why v7 exists:

- v5 optimized sparse triplets plus paragraph/repulsion losses, so it did not directly optimize corpus-level KNN preservation.
- v6 used a parametric t-SNE-like KL objective, but computed affinities inside each mini-batch, losing the global KNN graph used by the evaluator.
- v7 optimizes the same structure the evaluation cares about: high-dimensional semantic neighborhoods preserved in 2D.

Current corpus-level check:

```text
Ours(v7)  trust@12=0.906  knnOv@12=0.265
UMAP      trust@12≈0.898  knnOv@12≈0.261
```

For a more readable but slightly less faithful target, train a v8-style checkpoint
with a larger UMAP low-dimensional spacing:

```bash
python3 code_for_model/train_all_v7.py \
  --target umap \
  --target-neighbors 15 \
  --min-dist 0.20 \
  --spread 1.50 \
  --target-cache data/stages/05_model/semantic_target_umap_readable.npy \
  --epochs 300 \
  --out data/stages/05_model/bert2d_mapper_all_v8_readable.pt
```

This keeps the model semantic-first, but distills a less crowded layout target.

## 2. One-Command Pipeline

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

Note: `pipeline.py` still defaults to the v5 training command for historical compatibility. For the current v7 model, use the explicit v7 commands below.

## 3. Stage-By-Stage Flow

| Stage | Command | Input | Output |
|---|---|---|---|
| PDF to Markdown | `python3 code_for_data/mineru_pdf.py` | `data/pdf/**/*.pdf` | `data/stages/01_corpus/<paper>/<paper>.md` |
| Markdown to MSUs | `python3 code_for_data/build_corpus.py` | `data/stages/01_corpus/<paper>/<paper>.md` | `data/stages/02_msu/formdatabase.json` (+ `paragraphs.json`, `papers.json`) |
| Raw triplets | `python3 code_for_model/generate_triplets.py` | `data/stages/02_msu/formdatabase.json` | `data/stages/03_triplets/contrastive_triplets_raw.json` |
| Refined triplets | `python3 code_for_model/refine_triplets.py` | raw triplets + formdatabase | `data/stages/03_triplets/contrastive_triplets.json` |
| Embedding cache | `python3 code_for_model/precompute_embeddings.py` | `data/stages/02_msu/formdatabase.json` | `data/stages/04_embeddings/emb_corpus.npy` (+ `.ids.json`) |
| Mapper training | `python3 code_for_model/train_all_v7.py --epochs 300 --reuse-target --out data/stages/05_model/bert2d_mapper_all_v7_pure.pt` | BGE embedding cache | `data/stages/05_model/bert2d_mapper_all_v7_pure.pt` |
| 2D coordinates | `python3 code_for_data/formdatabase.py --input data/stages/02_msu/formdatabase.json --mapper data/stages/05_model/bert2d_mapper_all_v7_pure.pt --out data/stages/02_msu/formdatabase_v7.json` | formdatabase + v7 mapper | `data/stages/02_msu/formdatabase_v7.json` |
| Hex/HSU grouping | `python3 code_for_data/generate_hex.py` | `data/stages/02_msu/formdatabase_v2.0.json` | `data/stages/06_hex/hexagon_info.json` |
| HSU summaries | `python3 code_for_data/summarize_hex.py` | hex info + formdatabase_v2.0 | `data/stages/07_summaries/summaries.json` |
| Case files | `python3 code_for_data/build_case_files.py` | formdatabase_v2.0 + summaries | `data/outputs/case_generated/database-*.json`, `summary-*.json` |
| Frontend data | `python3 build_semantic_map_from_db_summary.py --case-dir data/outputs/case_generated --out data/stages/08_semantic_map/semantic_map_data.json` | case files | `data/stages/08_semantic_map/semantic_map_data.json` |

Current v7 corpus commands:

```bash
python3 code_for_model/train_all_v7.py \
  --epochs 300 \
  --reuse-target \
  --out data/stages/05_model/bert2d_mapper_all_v7_pure.pt

python3 code_for_data/formdatabase.py \
  --input data/stages/02_msu/formdatabase.json \
  --mapper data/stages/05_model/bert2d_mapper_all_v7_pure.pt \
  --out data/stages/02_msu/formdatabase_v7.json

python3 code_for_data/generate_hex.py \
  --input data/stages/02_msu/formdatabase_v7.json \
  --out data/stages/06_hex/hexagon_info_v7.json \
  --hex-size 0.15

python3 code_for_data/summarize_hex.py \
  --hex-info data/stages/06_hex/hexagon_info_v7.json \
  --formdb data/stages/02_msu/formdatabase_v7.json \
  --out data/stages/07_summaries/summaries_v7.json \
  --no-llm

python3 code_for_data/build_case_files.py \
  --formdb data/stages/02_msu/formdatabase_v7.json \
  --summaries data/stages/07_summaries/summaries_v7.json \
  --out-dir data/outputs/case_generated_v7_clean

python3 build_semantic_map_from_db_summary.py \
  --case-dir data/outputs/case_generated_v7_clean \
  --out data/stages/08_semantic_map/semantic_map_data_v7.json
```

Copy the generated map into the optional v7 project:

```bash
mkdir -p data/v7
cp data/stages/08_semantic_map/semantic_map_data_v7.json data/v7/semantic_map_data.json
```

The backend can then serve it with:

```text
/api/semantic-map?project_id=v7
```

## 4. Refresh Existing Frontend Cases With v7 Coordinates

The frontend's normal demo cases read:

```text
data/case1/semantic_map_data.json
data/case2/semantic_map_data.json
```

To update those existing cases after training a new mapper, first back them up:

```bash
ts=$(date +%Y%m%d_%H%M%S)
dest="data/archive/case1_case2_before_v7_${ts}"
mkdir -p "$dest"
cp -a data/case1 "$dest/case1"
cp -a data/case2 "$dest/case2"
```

Then apply the v7 mapper to each case's `database-*.json` files:

```bash
mkdir -p data/outputs/case_refresh_v7

python3 code_for_data/apply_mapper_to_case.py \
  --case-dir data/case1 \
  --out-dir data/outputs/case_refresh_v7/case1_db \
  --combined-out data/outputs/case_refresh_v7/formdatabase_case1_v7.json \
  --mapper data/stages/05_model/bert2d_mapper_all_v7_pure.pt

python3 code_for_data/apply_mapper_to_case.py \
  --case-dir data/case2 \
  --out-dir data/outputs/case_refresh_v7/case2_db \
  --combined-out data/outputs/case_refresh_v7/formdatabase_case2_v7.json \
  --mapper data/stages/05_model/bert2d_mapper_all_v7_pure.pt
```

Rebuild hexes, summaries, split case files, and frontend JSON:

```bash
# case1
python3 code_for_data/generate_hex.py \
  --input data/outputs/case_refresh_v7/formdatabase_case1_v7.json \
  --out data/outputs/case_refresh_v7/hexagon_info_case1_v7.json \
  --hex-size 0.15

python3 code_for_data/summarize_hex.py \
  --hex-info data/outputs/case_refresh_v7/hexagon_info_case1_v7.json \
  --formdb data/outputs/case_refresh_v7/formdatabase_case1_v7.json \
  --out data/outputs/case_refresh_v7/summaries_case1_v7.json \
  --no-llm

python3 code_for_data/build_case_files.py \
  --formdb data/outputs/case_refresh_v7/formdatabase_case1_v7.json \
  --summaries data/outputs/case_refresh_v7/summaries_case1_v7.json \
  --out-dir data/outputs/case_refresh_v7/case1_frontend

python3 build_semantic_map_from_db_summary.py \
  --case-dir data/outputs/case_refresh_v7/case1_frontend \
  --out data/outputs/case_refresh_v7/semantic_map_data_case1_v7.json

# case2: same commands with case2 paths.
```

After inspection, copy refreshed files into the live frontend case folders:

```bash
find data/case1 data/case2 -maxdepth 1 \
  \( -name 'database-*.json' -o -name 'summary-*.json' -o -name 'semantic_map_data.json' \) \
  -delete

cp data/outputs/case_refresh_v7/case1_frontend/database-*.json data/case1/
cp data/outputs/case_refresh_v7/case1_frontend/summary-*.json data/case1/
cp data/outputs/case_refresh_v7/semantic_map_data_case1_v7.json data/case1/semantic_map_data.json

cp data/outputs/case_refresh_v7/case2_frontend/database-*.json data/case2/
cp data/outputs/case_refresh_v7/case2_frontend/summary-*.json data/case2/
cp data/outputs/case_refresh_v7/semantic_map_data_case2_v7.json data/case2/semantic_map_data.json
```

For final demo text, rerun `summarize_hex.py` without `--no-llm` to replace fallback summaries with LLM-written HSU summaries.

## 5. Standard Data Contracts

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

### `formdatabase_v2.0.json` / `formdatabase_v7.json`

Same as `formdatabase.json`, plus:

```json
{ "2d_coord": [1.23, 4.56] }
```

### `hexagon_info*.json` / `summaries*.json`

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

## 6. Active vs Legacy Scripts

Active pipeline scripts live in:

- `code_for_data/mineru_pdf.py`
- `code_for_data/build_corpus.py`
- `code_for_model/generate_triplets.py`
- `code_for_model/refine_triplets.py`
- `code_for_model/precompute_embeddings.py`
- `code_for_model/train_all_v7.py`
- `code_for_data/formdatabase.py`
- `code_for_data/apply_mapper_to_case.py`
- `code_for_data/generate_hex.py`
- `code_for_data/summarize_hex.py`
- `code_for_data/build_case_files.py`
- `build_semantic_map_from_db_summary.py`
- `pipeline.py`

Old hard-coded or experimental scripts are archived under `legacy/`.

Legacy or diagnostic model scripts:

- `code_for_model/train_all_v5.py`
- `code_for_model/train_all_v6.py`
- `code_for_model/eval_faithfulness.py`
