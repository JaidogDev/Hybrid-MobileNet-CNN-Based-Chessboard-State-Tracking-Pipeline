import cv2, numpy as np, os, random, warnings
from .utils import ensure_dir
from .board import warp_board, split_grid

CLASSES = ["Empty","WP","WN","WB","WR","WQ","WK","BP","BN","BB","BR","BQ","BK"]
INITIAL_RANKS = [
    ['BR','BN','BB','BQ','BK','BB','BN','BR'],
    ['BP']*8, ['Empty']*8, ['Empty']*8,
    ['Empty']*8, ['Empty']*8, ['WP']*8,
    ['WR','WN','WB','WQ','WK','WB','WN','WR']
]
LABEL2IDX = {c:i for i,c in enumerate(CLASSES)}

def bootstrap_from_first_frame(video_path, out_dir, cfg):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    warped,_ = warp_board(frame, cfg)
    cells = split_grid(warped, cfg["cells"]["img_size"])

    saved = 0
    for (r,c), patch in cells:
        label = INITIAL_RANKS[r][c]
        cls_dir = os.path.join(out_dir, label)
        ensure_dir(cls_dir)
        fname = f"{os.path.splitext(os.path.basename(video_path))[0]}_{r}{c}.jpg"
        cv2.imwrite(os.path.join(cls_dir, fname), patch)
        saved += 1
    return saved

def simple_augment(img):
    warnings.warn(
        "simple_augment() is deprecated and will be removed in a future release. "
        "Use your TF/Keras augmentation pipeline instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), random.uniform(-15,15), 1.0)
    aug = cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)
    alpha = 1.0 + random.uniform(-0.2,0.2)  # contrast
    beta  = random.uniform(-20,20)          # brightness
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    if random.random() < 0.5:
        k = random.choice([3,5])
        aug = cv2.GaussianBlur(aug, (k,k), 0)
    if random.random() < 0.2:
        x0 = random.randint(0,w-1); y0 = random.randint(0,h-1)
        dx = random.randint(w//10, w//4); dy = random.randint(h//10, h//4)
        cv2.rectangle(aug, (x0,y0), (min(w-1,x0+dx),min(h-1,y0+dy)), (0,0,0), -1)
    return aug
