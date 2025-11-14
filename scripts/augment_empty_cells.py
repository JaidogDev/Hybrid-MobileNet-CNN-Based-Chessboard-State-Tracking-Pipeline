"""
Extract & Augment Empty cells from videos (Improved Quality)
"""
import cv2
import numpy as np
from pathlib import Path
import random
import sys

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from Chess_Detection_Competition.utils import load_config
from Chess_Detection_Competition.board import warp_board

# ===== CONFIGURATION =====
cfg = load_config()
VIDEOS_DIR = ROOT / cfg["paths"]["videos_dir"]
EMPTY_OUT = ROOT / cfg["paths"]["cells_bootstrap_dir"] / "Empty"
EMPTY_OUT.mkdir(parents=True, exist_ok=True)

# ===== HELPER FUNCTIONS =====

def is_empty_cell_strict(patch):
    """
    ตรวจสอบ Empty cell แบบเข้มงวด (Improved Quality)
    """
    # 1. ตรวจสอบสี (board pattern)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    
    green_lower = np.array([35, 30, 30])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    gray_lower = np.array([0, 0, 100])
    gray_upper = np.array([180, 30, 255])
    gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
    
    board_mask = cv2.bitwise_or(green_mask, gray_mask)
    board_ratio = board_mask.sum() / (patch.shape[0] * patch.shape[1] * 255)
    
    if board_ratio < 0.75:
        return False
    
    # 2. ตรวจสอบ edge
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = edges.sum() / (patch.shape[0] * patch.shape[1] * 255)
    
    if edge_ratio > 0.15:
        return False
    
    # 3. ตรวจสอบ variance
    variance = np.var(gray)
    if variance > 1000:
        return False
    
    return True


def extract_empty_from_video(video_path, out_dir, target_per_video=100):
    """
    Extract Empty cells from a video (middle-game positions)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    # Skip first 60 frames
    for _ in range(60):
        cap.grab()
    
    saved = 0
    frame_id = 60
    img_size = int(cfg["cells"]["img_size"])
    
    while cap.isOpened() and saved < target_per_video:
        ok = cap.grab()
        if not ok:
            break
        
        frame_id += 1
        
        if frame_id % 10 != 0:
            continue
        
        ok, frame = cap.retrieve()
        if not ok:
            break
        
        try:
            warped, _ = warp_board(frame, cfg)
            
            H, W = warped.shape[:2]
            cell_h = H // 8
            cell_w = W // 8
            
            for _ in range(8):
                r = random.randint(0, 7)
                c = random.randint(0, 7)
                
                y1 = r * cell_h
                y2 = (r + 1) * cell_h
                x1 = c * cell_w
                x2 = (c + 1) * cell_w
                
                patch = warped[y1:y2, x1:x2]
                patch = cv2.resize(patch, (img_size, img_size))
                
                if is_empty_cell_strict(patch):
                    fname = f"{video_path.stem}_f{frame_id}_r{r}c{c}.jpg"
                    cv2.imwrite(str(out_dir / fname), patch)
                    saved += 1
                    
                    if saved >= target_per_video:
                        break
        
        except Exception as e:
            continue
    
    cap.release()
    return saved


def augment_patch(img):
    """Apply random augmentation"""
    img = img.copy()
    
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    factor = random.uniform(0.8, 1.2)
    img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img


def augment_existing_cells(empty_dir, target=500):
    """Augment existing Empty cells to reach target count"""
    existing = list(empty_dir.glob("*.jpg"))
    
    if not existing:
        print("WARNING: No existing Empty cells found.")
        return 0
    
    print(f"Found {len(existing)} existing Empty cells")
    
    needed = target - len(existing)
    
    if needed <= 0:
        print("Already have enough Empty cells")
        return 0
    
    print(f"Generating {needed} augmented Empty cells...")
    
    for i in range(needed):
        src_path = random.choice(existing)
        img = cv2.imread(str(src_path))
        
        if img is None:
            continue
        
        aug = augment_patch(img)
        
        fname = f"aug_{i}_{src_path.name}"
        cv2.imwrite(str(empty_dir / fname), aug)
    
    final_count = len(list(empty_dir.glob("*.jpg")))
    print(f"Total Empty cells: {final_count}")
    return needed


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    print("=" * 50)
    print("Empty Cell Extraction & Augmentation")
    print("=" * 50)
    
    # ===== STEP 1: Extract from videos =====
    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    
    if not videos:
        print("❌ No videos found in", VIDEOS_DIR)
    else:
        print(f"\nFound {len(videos)} videos")
        print("Extracting Empty cells from middle-game...\n")
        
        total_saved = 0
        for v in videos:
            print(f"Processing: {v.name}")
            saved = extract_empty_from_video(v, EMPTY_OUT, target_per_video=100)
            print(f"  → Saved {saved} empty cells")
            total_saved += saved
        
        print(f"\nExtraction complete: {total_saved} cells")
    
    # ===== STEP 2: Augment to reach target =====
    print("\n" + "=" * 50)
    print("Augmentation Phase")
    print("=" * 50 + "\n")
    
    augmented = augment_existing_cells(EMPTY_OUT, target=500)
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Output directory: {EMPTY_OUT}")
    print(f"Total Empty cells: {len(list(EMPTY_OUT.glob('*.jpg')))}")
    print("=" * 50)

# Cell: รัน augment_empty_cells.py
import sys
from pathlib import Path

# เปลี่ยน working directory ไป scripts/
scripts_dir = Path("../scripts").resolve()
sys.path.insert(0, str(scripts_dir.parent))

print(f"Running augment_empty_cells.py from {scripts_dir}\n")

# Import และรัน
import importlib.util
spec = importlib.util.spec_from_file_location("augment", scripts_dir / "augment_empty_cells.py")
augment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augment)

print("\nScript completed!")