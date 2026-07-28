#!/bin/bash
# 用干净的最终方法(v11: 邻域保持投影 + LLM 对应监督)在基因组语料上重建 case3 前端数据。
set -e
cd /Users/yudong/Desktop/SeSMap/SeSMap-backend
export SESMAP_STAGES_DIR=data/bio_eval/stages
S=$SESMAP_STAGES_DIR
mkdir -p $S/04_embeddings $S/05_model $S/06_hex $S/07_summaries

echo "########## 0. 备份 case3 前端数据(保留 gallery/thumbnails) ##########"
BK=data/case3/backup_pre_v11; mkdir -p $BK
cp data/case3/database-*.json data/case3/summary-*.json data/case3/semantic_map_data.json $BK/ 2>/dev/null || true

echo "########## 1. bge 句向量 ##########"
python3 code_for_model/precompute_embeddings.py --corpus $S/02_msu/formdatabase.json \
    --out $S/04_embeddings/emb_corpus.npy --id-field idx --sentence-field sentence

echo "########## 2. LLM 跨论文语义对应对 ##########"
python3 code_for_data/augment_pairs_llm.py --corpus $S/02_msu/formdatabase.json \
    --emb $S/04_embeddings/emb_corpus.npy --out $S/02_msu/llm_pairs.json \
    --max-candidates 3000 --judge-batch 10

echo "########## 3. 训练基因组 v11 (t-SNE 邻域保持 + 对应监督 λ=0.1) ##########"
python3 code_for_model/train_contrastive_v11.py --cache $S/04_embeddings/emb_corpus.npy \
    --pairs $S/02_msu/llm_pairs.json --lambda-tsne 1 --lambda-con 0.1 --out $S/05_model/v11_genomics.pt

echo "########## 4. 投影 -> 2d_coord ##########"
python3 code_for_data/formdatabase.py --input $S/02_msu/formdatabase.json \
    --out $S/02_msu/formdatabase_v2.0.json --mapper $S/05_model/v11_genomics.pt

echo "########## 5. 坐标缩放(标准化 x1.8,对齐前端 hex 尺度) ##########"
python3 - "$S" <<'PY'
import json, numpy as np, sys
S=sys.argv[1]
d=json.load(open(f"{S}/02_msu/formdatabase_v2.0.json"))
C=np.array([r["2d_coord"] for r in d if "2d_coord" in r], dtype=float)
mu=C.mean(0); sd=C.std(0); sd[sd<1e-9]=1
for r in d:
    if "2d_coord" in r:
        r["2d_coord"]=((np.array(r["2d_coord"],dtype=float)-mu)/sd*1.8).tolist()
json.dump(d, open(f"{S}/02_msu/formdatabase_v2.0.json","w"), ensure_ascii=False, indent=2)
print("rescaled to std~1.8")
PY

echo "########## 6. hex -> summary -> case files -> 剔除 None/Other -> semantic_map ##########"
python3 code_for_data/generate_hex.py
python3 code_for_data/summarize_hex.py --no-llm
python3 code_for_data/build_case_files.py --out-dir data/case3
# 只保留 5 个 discourse-role 子空间,剔除 None/Other 等
for r in None none Other others Others; do
  rm -f "data/case3/database-$r.json" "data/case3/summary-$r.json" 2>/dev/null || true
done
python3 build_semantic_map_from_db_summary.py --case-dir data/case3 --out data/case3/semantic_map_data.json

echo "########## DONE ##########"
ls data/case3/database-*.json
python3 -c "import json;d=json.load(open('data/case3/semantic_map_data.json'));print('subspaces:',len(d.get('subspaces',[])) if isinstance(d,dict) else 'n/a')"
