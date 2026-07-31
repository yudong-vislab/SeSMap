#!/bin/bash
# 增量更新 case3：仅为新增 PDF 生成 Markdown/MSU，随后用全部论文重建地图与图库。
set -euo pipefail

cd /Users/yudong/Desktop/SeSMap/SeSMap-backend
export SESMAP_STAGES_DIR=data/bio_eval/stages
S=$SESMAP_STAGES_DIR
CASE=case3

echo "########## 0. Backup current case3 frontend data ##########"
BK="data/$CASE/backup_before_incremental_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BK"
cp data/$CASE/database-*.json data/$CASE/summary-*.json data/$CASE/semantic_map_data.json data/$CASE/gallery.json "$BK"/ 2>/dev/null || true

echo "########## 1. New PDFs -> Markdown only ##########"
python3 code_for_data/mineru_pdf.py \
  --pdf-dir data/$CASE/pdf --corpus "$S/01_corpus" --skip-existing

echo "########## 2. New Markdown -> MSU only; reuse existing MSUs ##########"
python3 code_for_data/build_corpus.py --corpus "$S/01_corpus" --reuse-existing

echo "########## 2b. Conservative Markdown-quality MSU filtering ##########"
cp "$S/02_msu/formdatabase.json" "$S/02_msu/formdatabase_raw.json"
python3 code_for_data/filter_msu_quality.py \
  --input "$S/02_msu/formdatabase_raw.json" --out "$S/02_msu/formdatabase.json" \
  --report "$S/02_msu/msu_quality_report.json"

echo "########## 3. Re-embed complete 12-paper corpus ##########"
python3 code_for_model/precompute_embeddings.py \
  --corpus "$S/02_msu/formdatabase.json" --out "$S/04_embeddings/emb_corpus.npy" \
  --id-field idx --sentence-field sentence

echo "########## 4. Cross-paper semantic supervision ##########"
# Every MSU retained by the conservative Markdown-quality filter is covered by
# at least one submitted cross-paper pair.  Do not revert this to a small
# random candidate cap: the case3 layout must be trained from the full retained
# corpus, not a semantic/rank subset.
python3 code_for_data/augment_pairs_llm.py \
  --corpus "$S/02_msu/formdatabase.json" --emb "$S/04_embeddings/emb_corpus.npy" \
  --out "$S/02_msu/llm_pairs.json" \
  --tau-lo 0.35 --tau-hi 0.98 --per-anchor 1 --max-candidates 6000 --judge-batch 20

echo "########## 5. Train v11 projection ##########"
python3 code_for_model/train_contrastive_v11.py \
  --cache "$S/04_embeddings/emb_corpus.npy" --pairs "$S/02_msu/llm_pairs.json" \
  --lambda-tsne 1 --lambda-con 0.1 --out "$S/05_model/v11_genomics.pt"

echo "########## 6. Project and normalize 2D coordinates ##########"
python3 code_for_data/formdatabase.py --input "$S/02_msu/formdatabase.json" \
  --out "$S/02_msu/formdatabase_v2.0.json" --mapper "$S/05_model/v11_genomics.pt"
python3 - "$S" <<'PY'
import json, numpy as np, sys
s = sys.argv[1]
p = f"{s}/02_msu/formdatabase_v2.0.json"
d = json.load(open(p, encoding="utf-8"))
c = np.array([r["2d_coord"] for r in d if "2d_coord" in r], dtype=float)
mu, sd = c.mean(0), c.std(0)
sd[sd < 1e-9] = 1
for r in d:
    if "2d_coord" in r:
        r["2d_coord"] = ((np.asarray(r["2d_coord"], dtype=float) - mu) / sd * 1.8).tolist()
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"normalized {len(c)} MSUs")
PY

echo "########## 7. Hexes, frontend case files, and gallery ##########"
python3 code_for_data/generate_hex.py
python3 code_for_data/summarize_hex.py --no-llm
find data/$CASE -maxdepth 1 -type f \( -name 'database-*.json' -o -name 'summary-*.json' \) -delete
python3 code_for_data/build_case_files.py --out-dir data/$CASE
# Keep the five discourse-role panels used by the existing case3 UI.
rm -f data/$CASE/database-Other.json data/$CASE/summary-Other.json
python3 build_semantic_map_from_db_summary.py --case-dir data/$CASE --out data/$CASE/semantic_map_data.json
python3 code_for_data/extract_thumbnails.py --corpus "$S/01_corpus" \
  --papers "$S/02_msu/papers.json" --case $CASE --source-offset 8

echo "########## DONE ##########"
python3 - <<'PY'
import json
d = json.load(open("data/case3/semantic_map_data.json", encoding="utf-8"))
g = json.load(open("data/case3/gallery.json", encoding="utf-8"))
print("MSUs:", d["stats"]["totals"]["msu_count"], "gallery papers:", len(g))
PY
