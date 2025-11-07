# scripts/bootstrap_from_videos.py
import glob, os
from src.utils import load_config, get_logger
from src.cells import bootstrap_from_first_frame

cfg = load_config()
log = get_logger()

videos = glob.glob(os.path.join(cfg["paths"]["videos_dir"], "*.mp4"))
out_dir = cfg["paths"]["cells_bootstrap_dir"]

total = 0
for v in videos:
    saved = bootstrap_from_first_frame(v, out_dir, cfg)
    log.info(f"{os.path.basename(v)} -> {saved} patches")
    total += saved
log.info(f"Total patches: {total}")
