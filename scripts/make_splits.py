import os, glob, shutil, random
from Chess_Detection_Competition.utils import load_config, ensure_dir, get_logger

cfg = load_config(); log = get_logger()
src_dirs = [cfg["paths"]["cells_bootstrap_dir"], cfg["paths"]["cells_public_dir"]]
dst_train = cfg["paths"]["final_train_dir"]
dst_val   = cfg["paths"]["final_val_dir"]
VAL_RATIO = cfg["train"]["val_split"]

def collect(root):
    items = []
    if not os.path.isdir(root): return items
    for cls in os.listdir(root):
        p = os.path.join(root, cls)
        if not os.path.isdir(p): continue
        for f in glob.glob(os.path.join(p, "*.jpg")):
            items.append((f, cls))
    return items

all_items = []
for d in src_dirs:
    all_items += collect(d)

random.shuffle(all_items)
n_val = int(len(all_items) * VAL_RATIO)
val_items = all_items[:n_val]
train_items = all_items[n_val:]

def copy(items, base):
    for f, cls in items:
        dst = os.path.join(base, cls)
        ensure_dir(dst)
        shutil.copy2(f, os.path.join(dst, os.path.basename(f)))

copy(train_items, dst_train)
copy(val_items,   dst_val)
log.info(f"Train: {len(train_items)}  Val: {len(val_items)}")
