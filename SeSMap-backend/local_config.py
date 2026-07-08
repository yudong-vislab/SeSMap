"""
Central local paths for the SeSMap data/model pipeline.
所有流水线脚本从这里取路径；任何一项都可用 .env 里的同名环境变量覆盖。

目录归档结构（v2）：
  data/
  ├── pdf/                     训练主语料的输入 PDF
  ├── stages/                  训练主语料的“分阶段”产物（一个阶段一个文件夹）
  │   ├── 01_corpus/           build_corpus 的 md/sections 中间产物
  │   ├── 02_msu/              formdatabase.json (+ _v2.0 加坐标) + paragraphs/papers
  │   ├── 03_triplets/         contrastive_triplets_raw.json + contrastive_triplets.json
  │   ├── 04_embeddings/       emb_corpus.npy (+ .ids.json)
  │   ├── 05_model/            bert2d_mapper_all_v5.pt (+ 检查点/loss 图)
  │   ├── 06_hex/              hexagon_info.json
  │   ├── 07_summaries/        summaries.json
  │   └── 08_semantic_map/     semantic_map_data.json（训练主语料的前端数据）
  ├── case1/  case2/  case3/   每个 case 独立归档：semantic_map_data.json + database-*/summary-* + pdf/
  ├── outputs/                 杂项输出（评测结果/缓存、case_generated 等）
  └── archive/                 旧文件/备份
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

BACKEND_DIR = Path(__file__).resolve().parent


def _p(env_name: str, default: Path) -> Path:
    v = os.getenv(env_name)
    return Path(v).expanduser() if v else default


# --- 本地 bge 编码器（论文用 bge-large-en-v1.5, 1024 维） ---
BGE_MODEL_PATH = _p("BGE_MODEL_PATH", BACKEND_DIR / "models" / "bge-large-en-v1.5")

# --- 数据根 ---
DATA_ROOT   = _p("SESMAP_DATA_ROOT",   BACKEND_DIR / "data")
PDF_DIR     = _p("SESMAP_PDF_DIR",     DATA_ROOT / "pdf")        # 训练主语料输入 PDF
OUTPUT_DIR  = _p("SESMAP_OUTPUT_DIR",  DATA_ROOT / "outputs")    # 杂项输出（评测结果/缓存等）
ARCHIVE_DIR = _p("SESMAP_ARCHIVE_DIR", DATA_ROOT / "archive")    # 旧文件/备份
CASE_ROOT   = _p("SESMAP_CASE_ROOT",   DATA_ROOT)                # 各 case 归档在 CASE_ROOT/caseN（app 读 data/caseN）

# --- 训练主语料的分阶段目录 ---
STAGES_DIR = _p("SESMAP_STAGES_DIR", DATA_ROOT / "stages")
CORPUS_DIR = STAGES_DIR / "01_corpus"        # build_corpus 中间（<name>/<name>.md /.json /_rewrite.json）
MSU_DIR    = STAGES_DIR / "02_msu"
TRIPLET_DIR = STAGES_DIR / "03_triplets"
EMB_DIR    = STAGES_DIR / "04_embeddings"
MODEL_DIR  = STAGES_DIR / "05_model"
HEX_DIR    = STAGES_DIR / "06_hex"
SUM_DIR    = STAGES_DIR / "07_summaries"
SMAP_DIR   = STAGES_DIR / "08_semantic_map"

# --- 各阶段标准产物文件名 ---
FORMDB       = MSU_DIR / "formdatabase.json"          # MSU 库（无坐标）
FORMDB_V2    = MSU_DIR / "formdatabase_v2.0.json"     # 加了 2d_coord
TRIPLETS_RAW = TRIPLET_DIR / "contrastive_triplets_raw.json"
TRIPLETS     = TRIPLET_DIR / "contrastive_triplets.json"
EMB_CACHE    = EMB_DIR / "emb_corpus.npy"
MAPPER_CKPT  = _p("SESMAP_MAPPER_CKPT", MODEL_DIR / "bert2d_mapper_all_v5.pt")
HEX_INFO     = HEX_DIR / "hexagon_info.json"
SUMMARIES    = SUM_DIR / "summaries.json"
SEMANTIC_MAP = SMAP_DIR / "semantic_map_data.json"

# --- 由 formdatabase_v2.0 + summaries 切分出的 case 生成物 ---
CASE_BUILD_DIR = _p("SESMAP_CASE_BUILD_DIR", OUTPUT_DIR / "case_generated")

_STAGE_DIRS = [CORPUS_DIR, MSU_DIR, TRIPLET_DIR, EMB_DIR, MODEL_DIR, HEX_DIR, SUM_DIR, SMAP_DIR]


def ensure_dirs():
    for d in (DATA_ROOT, PDF_DIR, OUTPUT_DIR, ARCHIVE_DIR, STAGES_DIR,
              *_STAGE_DIRS, CASE_BUILD_DIR, BGE_MODEL_PATH.parent):
        Path(d).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    for k in ["BACKEND_DIR", "BGE_MODEL_PATH", "DATA_ROOT", "PDF_DIR", "STAGES_DIR",
              "CORPUS_DIR", "MSU_DIR", "FORMDB", "FORMDB_V2", "TRIPLET_DIR", "TRIPLETS_RAW",
              "TRIPLETS", "EMB_CACHE", "MAPPER_CKPT", "HEX_INFO", "SUMMARIES", "SEMANTIC_MAP",
              "CASE_ROOT", "CASE_BUILD_DIR", "OUTPUT_DIR", "ARCHIVE_DIR"]:
        print(f"{k:16} = {globals()[k]}")
    print("bge_exists       =", BGE_MODEL_PATH.exists())
    for name, p in [("FORMDB", FORMDB), ("TRIPLETS", TRIPLETS), ("EMB_CACHE", EMB_CACHE)]:
        print(f"{name}_exists  = {Path(p).exists()}")
