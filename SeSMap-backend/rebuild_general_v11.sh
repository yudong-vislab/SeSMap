#!/bin/bash
# 训练一个新的通用 v11 模型（case1+case2 合并语料，二者都在样本内），
# 再用它给 case1/case2 重算布局坐标并写成前端数据结构。
set -e
cd /Users/yudong/Desktop/SeSMap/SeSMap-backend
export SESMAP_STAGES_DIR=data/general_v11/stages
G=$SESMAP_STAGES_DIR
mkdir -p $G/02_msu $G/04_embeddings $G/05_model

echo "########## 1. 合并 case1+case2 语料（全局唯一 paper_id 供训练；保留原字段供渲染） ##########"
python3 - <<'PY'
import json, glob
combined=[]; gp=0
for case in ["case1","case2"]:
    recs=[]
    for f in sorted(glob.glob(f"data/{case}/database-*.json")):
        recs += json.load(open(f, encoding="utf-8"))
    locals_=sorted(set(r.get("paper_id") for r in recs))
    remap={lp: gp+i for i,lp in enumerate(locals_)}; gp += len(locals_)
    for r in recs:
        r=dict(r); r["orig_paper_id"]=r.get("paper_id"); r["case"]=case
        r["paper_id"]=remap[r.get("paper_id")]
        combined.append(r)
for i,r in enumerate(combined): r["idx"]=i
json.dump(combined, open("data/general_v11/stages/02_msu/formdatabase.json","w"), ensure_ascii=False, indent=2)
print(f"combined {len(combined)} MSUs, {gp} global papers (case1+case2)")
PY

echo "########## 2. bge 句向量 ##########"
python3 code_for_model/precompute_embeddings.py --corpus $G/02_msu/formdatabase.json \
    --out $G/04_embeddings/emb_corpus.npy --id-field idx --sentence-field sentence

echo "########## 3. LLM 跨论文语义等价对 ##########"
python3 code_for_data/augment_pairs_llm.py --corpus $G/02_msu/formdatabase.json \
    --emb $G/04_embeddings/emb_corpus.npy --out $G/02_msu/llm_pairs.json \
    --max-candidates 4000 --judge-batch 10

echo "########## 4. 训练通用 v11 (t-SNE + 语义监督 λ=0.1) ##########"
python3 code_for_model/train_contrastive_v11.py --cache $G/04_embeddings/emb_corpus.npy \
    --pairs $G/02_msu/llm_pairs.json --lambda-tsne 1 --lambda-con 0.3 --out $G/05_model/v11_general.pt

echo "########## 5. 用通用模型投影合并语料 ##########"
python3 code_for_data/formdatabase.py --input $G/02_msu/formdatabase.json \
    --out $G/02_msu/formdatabase_v2.0.json --mapper $G/05_model/v11_general.pt

echo "########## 6. 拆回各 case + 逐 case 缩放 + 重建前端数据 ##########"
for CASE in case1 case2; do
  export SESMAP_STAGES_DIR=data/$CASE/stages
  S=$SESMAP_STAGES_DIR
  mkdir -p $S/02_msu $S/06_hex $S/07_summaries
  BK=data/$CASE/backup_pre_v11; mkdir -p $BK
  cp data/$CASE/database-*.json data/$CASE/summary-*.json data/$CASE/semantic_map_data.json $BK/ 2>/dev/null || true
  python3 - "$CASE" "$G" "$S" <<'PY'
import json, numpy as np, sys
case, G, S = sys.argv[1], sys.argv[2], sys.argv[3]
comb=json.load(open(f"{G}/02_msu/formdatabase_v2.0.json"))
sub=[r for r in comb if r.get("case")==case]
C=np.array([r["2d_coord"] for r in sub if "2d_coord" in r], dtype=float)
mu=C.mean(0); sd=C.std(0); sd[sd<1e-9]=1
out=[]
for r in sub:
    r=dict(r); r["paper_id"]=r.pop("orig_paper_id", r.get("paper_id")); r.pop("case",None)
    if "2d_coord" in r:
        r["2d_coord"]=((np.array(r["2d_coord"],dtype=float)-mu)/sd*1.8).tolist()
    out.append(r)
json.dump(out, open(f"{S}/02_msu/formdatabase_v2.0.json","w"), ensure_ascii=False, indent=2)
print(f"[{case}] {len(out)} MSUs rendered from general v11 model")
PY
  python3 code_for_data/generate_hex.py
  python3 code_for_data/summarize_hex.py --no-llm
  python3 code_for_data/build_case_files.py --out-dir data/$CASE
  python3 build_semantic_map_from_db_summary.py --case-dir data/$CASE --out data/$CASE/semantic_map_data.json
  echo "  [$CASE] semantic_map rebuilt"
done
echo "########## DONE ##########"
