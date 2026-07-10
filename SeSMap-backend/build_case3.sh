#!/bin/bash
# 用生信语料(bio_eval)构建 case3 前端数据，对齐 case1/2 契约。全部确定性、无 LLM。
set -e
cd /Users/yudong/Desktop/SeSMap/SeSMap-backend
export SESMAP_STAGES_DIR=/Users/yudong/Desktop/SeSMap/SeSMap-backend/data/bio_eval/stages
S=$SESMAP_STAGES_DIR

echo "########## 0. 备份旧 case3 + 补 paper_info/paragraph_info ##########"
[ -d data/case3 ] && mv data/case3 data/archive/case3_old_pre_genomics 2>/dev/null || true
mkdir -p data/case3
python3 - <<'PY'
import json, glob, numpy as np
S="data/bio_eval/stages/02_msu"
papers=json.load(open(f"{S}/papers.json"))
paras=json.load(open(f"{S}/paragraphs.json"))
d=json.load(open(f"{S}/formdatabase_v2.0.json"))
# 补 paper_info/paragraph_info（对齐 case2 schema，前端 gallery 用）
for r in d:
    pid=r.get("paper_id")
    if isinstance(pid,int) and 0<=pid<len(papers): r["paper_info"]=papers[pid]
    pa=r.get("para_id")
    if isinstance(pa,int) and 0<=pa<len(paras): r["paragraph_info"]=paras[pa]
# 关键：把 v10 的大尺度坐标(±120)缩放到 case2 尺度(±1.8)，否则前端 hex 网格过大只显示几格
c2=[]
for f in glob.glob("data/case2/database-*.json"):
    c2+=[x["2d_coord"] for x in json.load(open(f)) if "2d_coord" in x]
c2=np.array(c2); tgt_mean=c2.mean(0); tgt_std=c2.std(0)
C=np.array([r["2d_coord"] for r in d if "2d_coord" in r]); mu=C.mean(0); sd=C.std(0); sd[sd<1e-9]=1
for r in d:
    if "2d_coord" in r:
        r["2d_coord"]=(((np.array(r["2d_coord"])-mu)/sd)*tgt_std+tgt_mean).tolist()
json.dump(d,open(f"{S}/formdatabase_v2.0.json","w"),ensure_ascii=False,indent=2)
print(f"enriched + rescaled {len(d)} records to case2 coord scale")
PY

echo "########## 1. generate_hex ##########"
python3 code_for_data/generate_hex.py

echo "########## 2. summarize_hex (--no-llm 兜底) ##########"
python3 code_for_data/summarize_hex.py --no-llm

echo "########## 3. build_case_files -> data/case3 ##########"
python3 code_for_data/build_case_files.py --out-dir data/case3

echo "########## 4. build_semantic_map -> data/case3/semantic_map_data.json ##########"
python3 build_semantic_map_from_db_summary.py --case-dir data/case3 --out data/case3/semantic_map_data.json

echo "########## 5. PDFs -> data/pdf/case3 (供 gallery/RAG) ##########"
mkdir -p data/pdf/case3 && cp data/pdf/*.pdf data/pdf/case3/ 2>/dev/null || true
ls data/pdf/case3/ | wc -l | xargs echo "case3 PDFs:"

echo "########## 6. gallery 缩略图 + manifest(方案B,自动选每篇主图) ##########"
python3 code_for_data/extract_thumbnails.py \
    --corpus $S/01_corpus \
    --papers $S/02_msu/papers.json \
    --case case3 --source-offset 8

echo "########## DONE ##########"
ls data/case3/
