# src/Chess_Detection_Competition/utils.py
from pathlib import Path
import yaml, logging, os

def _project_root() -> Path:
    # utils.py -> (pkg) -> src -> <ROOT>
    return Path(__file__).resolve().parents[2]

def ensure_dir(p) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def get_logger(name: str = "chess", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # ป้องกันการติดซ้ำหลาย handler เวลารันใน notebook
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger

def load_config(path: str = "configs/parameters.yaml"):
    p = Path(path)
    full_path = p if p.is_absolute() else (_project_root() / p)
    if not full_path.exists():
        raise FileNotFoundError(f"Config not found: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
