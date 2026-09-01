#!/bin/bash
# 用最终模型 v11(parametric t-SNE + LLM 语义监督, λ=0.1)重建某个 case 的前端数据。
# 用法: bash build_case_v11.sh case2
set -e
cd /Users/yudong/Desktop/SeSMap/SeSMap-backend
CASE=$1
[ -z "$CASE" ] && { echo "usage: bash build_case_v11.sh <caseN>"; exit 1; }
export SESMAP_STAGES_DIR=data/$CASE/stages
S=$SESMAP_STAGES_DIR
mkdir -p $S/02_msu $S/04_embeddings $S/05_model $S/06_hex $S/07_summaries

echo "########## [$CASE] 0. 备份旧前端数据 + 从 database-*.json 重建语料 ##########"
BK=data/$CASE/backup_pre_v11
mkdir -p $BK
cp data/$CASE/database-*.json data/$CASE/summary-*.json data/$CASE/semantic_map_data.json $BK/ 2>/dev/null || true
python3 - "$CASE" "$S" <<'PY'
import json, glob, os, sys
case, S = sys.argv[1], sys.argv[2]
recs=[]
for f in sorted(glob.glob(f"data/{case}/database-*.json")):
    recs += json.load(open(f, encoding="utf-8"))
out=[]
for i,r in enumerate(recs):
    r=dict(r); r["idx"]=i
    out.append(r)
os.makedirs(f"{S}/02_msu", exist_ok=True)
json.dump(out, open(f"{S}/02_msu/formdatabase.json","w"), ensure_ascii=False, indent=2)
papers={}
for r in out:
    pid=r.get("paper_id"); pi=r.get("paper_info")
    if pid is not None and pi is not None and pid not in papers: papers[pid]=pi
json.dump([papers[k] for k in sorted(papers)], open(f"{S}/02_msu/papers.json","w"), ensure_ascii=False, indent=2)
print(f"[{case}] reconstructed {len(out)} MSUs, {len(papers)} papers")
PY

echo "########## [$CASE] 1. bge 句向量 ##########"
python3 code_for_model/precompute_embeddings.py --corpus $S/02_msu/formdatabase.json \
    --out $S/04_embeddings/emb_corpus.npy --id-field idx --sentence-field sentence

echo "########## [$CASE] 2. LLM 跨论文语义等价对 ##########"
python3 code_for_data/augment_pairs_llm.py --corpus $S/02_msu/formdatabase.json \
    --emb $S/04_embeddings/emb_corpus.npy --out $S/02_msu/llm_pairs.json \
    --max-candidates 3000 --judge-batch 10

echo "########## [$CASE] 3. 训练 v11 (t-SNE + 语义监督 λ=0.1) ##########"
python3 code_for_model/train_contrastive_v11.py --cache $S/04_embeddings/emb_corpus.npy \
    --pairs $S/02_msu/llm_pairs.json --lambda-tsne 1 --lambda-con 0.3 --out $S/05_model/v11.pt

echo "########## [$CASE] 4. 投影 -> 2d_coord ##########"
python3 code_for_data/formdatabase.py --input $S/02_msu/formdatabase.json \
    --out $S/02_msu/formdatabase_v2.0.json --mapper $S/05_model/v11.pt

echo "########## [$CASE] 5. 坐标缩放到显示尺度(标准化 x1.8) ##########"
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

echo "########## [$CASE] 6. hex -> summary(--no-llm) -> case files -> semantic_map ##########"
python3 code_for_data/generate_hex.py
python3 code_for_data/summarize_hex.py --no-llm
python3 code_for_data/build_case_files.py --out-dir data/$CASE
python3 build_semantic_map_from_db_summary.py --case-dir data/$CASE --out data/$CASE/semantic_map_data.json

echo "########## [$CASE] DONE ##########"
python3 -c "import json;d=json.load(open('data/$CASE/semantic_map_data.json'));print('subspaces:',len(d.get('subspaces',[])) if isinstance(d,dict) else 'n/a')"
