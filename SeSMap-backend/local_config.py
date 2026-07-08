"""
Central local paths for the SeSMap data/model pipeline.
所有流水线脚本从这里取路径；任何一项都可用 .env 里的同名环境变量覆盖。
把服务器上散落的 /home/lxy/... 与 pollution_result/ 统一收敛到这里。
"""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent


def _p(env_name: str, default: Path) -> Path:
    v = os.getenv(env_name)
    return Path(v).expanduser() if v else default


# --- 本地 bge 编码器（论文用 bge-large-en-v1.5, 1024 维；已下载到本地） ---
BGE_MODEL_PATH = _p("BGE_MODEL_PATH", BACKEND_DIR / "models" / "bge-large-en-v1.5")

# --- 数据根目录：所有输入/中间产物都在这下面 ---
DATA_ROOT  = _p("SESMAP_DATA_ROOT",   BACKEND_DIR / "data")
PDF_DIR    = _p("SESMAP_PDF_DIR",     DATA_ROOT / "pdf")       # 输入 PDF
CORPUS_DIR = _p("SESMAP_CORPUS_DIR",  DATA_ROOT / "corpus")    # 每篇论文一个子目录: <name>/<name>.md /.json /_rewrite.json
OUTPUT_DIR = _p("SESMAP_OUTPUT_DIR",  DATA_ROOT / "outputs")   # 汇总产物

# --- 训练好的投影模型（默认 v5） ---
MAPPER_CKPT = _p("SESMAP_MAPPER_CKPT", OUTPUT_DIR / "bert2d_mapper_all_v5.pt")

# --- 汇总产物的标准文件名 ---
FORMDB    = OUTPUT_DIR / "formdatabase.json"        # MSU 库（无坐标）
FORMDB_V2 = OUTPUT_DIR / "formdatabase_v2.0.json"   # 加了 2d_coord
TRIPLETS  = OUTPUT_DIR / "contrastive_triplets.json"
EMB_CACHE = OUTPUT_DIR / "emb_corpus.npy"


def ensure_dirs():
    for d in (DATA_ROOT, PDF_DIR, CORPUS_DIR, OUTPUT_DIR, BGE_MODEL_PATH.parent):
        Path(d).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    for k in ["BACKEND_DIR", "BGE_MODEL_PATH", "DATA_ROOT", "PDF_DIR",
              "CORPUS_DIR", "OUTPUT_DIR", "MAPPER_CKPT"]:
        print(f"{k:16} = {globals()[k]}")
    print("bge_exists       =", BGE_MODEL_PATH.exists())
