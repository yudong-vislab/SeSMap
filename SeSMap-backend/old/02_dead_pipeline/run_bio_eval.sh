#!/bin/bash
# 生信(基因组可视化 5 篇)标准建图 + 评测，全部隔离在 data/bio_eval/。
# 标准 = 每个语料建自己的忠实地图(生信训生信的 v10)，再报该图的 in-sample 保真；
# 附带一份"污染域模型迁移到生信"的诚实对照。
set -e
cd /Users/yudong/Desktop/SeSMap/SeSMap-backend
export SESMAP_STAGES_DIR=/Users/yudong/Desktop/SeSMap/SeSMap-backend/data/bio_eval/stages
POLLUTION_MODEL=/Users/yudong/Desktop/SeSMap/SeSMap-backend/data/stages/05_model/bert2d_mapper_all_v10.pt
S=$SESMAP_STAGES_DIR

echo "########## ① MinerU: case3 新增 PDF -> markdown ##########"
python3 code_for_data/mineru_pdf.py --pdf-dir data/case3/pdf --corpus $S/01_corpus --skip-existing

echo "########## ② build_corpus: 仅新增 markdown -> MSU (LLM 抽取) ##########"
python3 code_for_data/build_corpus.py --corpus $S/01_corpus --reuse-existing

echo "########## ③ precompute: bge 向量缓存 ##########"
python3 code_for_model/precompute_embeddings.py

echo "########## ④ 训练生信自己的 v10 图(自建图) ##########"
python3 code_for_model/train_all_v10.py

echo "########## ⑤a 生信坐标(生信模型, in-sample 地图) ##########"
python3 code_for_data/formdatabase.py

echo "########## ⑤b 生信坐标(污染域模型, 迁移对照) ##########"
python3 code_for_data/formdatabase.py --mapper $POLLUTION_MODEL --out $S/02_msu/formdatabase_transfer.json

echo "########## ⑥ 必要评测: 生信自建图 vs PCA/tSNE/UMAP (in-sample) ##########"
python3 code_for_model/eval_faithfulness.py --candidate-label "Ours(bio-self)" --v3 ""

echo "########## ⑦ 诚实对照: 污染模型迁移到生信 vs UMAP.transform vs fresh ##########"
python3 code_for_model/eval_domain_oos.py \
    --domain-formdb $S/02_msu/formdatabase_transfer.json \
    --domain-cache  $S/04_embeddings/emb_corpus.npy \
    --out data/bio_eval/domain_transfer_eval.json

echo "########## DONE ##########"
python3 - <<'PY'
import json
d=json.load(open("data/bio_eval/domain_transfer_eval.json"))
print("[迁移对照]", json.dumps(d["results"], ensure_ascii=False))
PY
