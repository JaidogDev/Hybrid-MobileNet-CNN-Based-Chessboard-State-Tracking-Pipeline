import cv2, numpy as np, os, random, warnings
from .utils import ensure_dir
from .board import warp_board, split_grid
from pathlib import Path
import chess

def bootstrap_from_first_frame(video_path, out_dir, cfg, start_fen=None):
    """
    Extract cells from first video frame (including Empty cells)
    """
    
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    
    if not ok or frame is None:
        print(f"[skip] cannot read {video_path}")
        return 0
    
    # Warp board
    try:
        from Chess_Detection_Competition.improved_board import warp_board_v2 as warp_board
    except:
        from Chess_Detection_Competition.board import warp_board
    
    warped, _ = warp_board(frame, cfg, manual_mode=False)
    
    # Split into 8x8
    try:
        from Chess_Detection_Competition.improved_board import split_grid_v2 as split_grid
    except:
        from Chess_Detection_Competition.board import split_grid
    
    img_size = int(cfg["cells"]["img_size"])
    cells = split_grid(warped, img_size)
    
    # 2. ใช้ FEN ที่เราส่งเข้ามา
    if start_fen:
        board = chess.Board(start_fen) # ใช้ FEN ที่กำหนด
    else:
        board = chess.Board() # ใช้ตำแหน่งเริ่มต้นมาตรฐาน (Fallback)
    
    saved_count = 0
    out_path = Path(out_dir)
    
    for (r, c), patch in cells:
        # Convert (r,c) → chess square
        file_idx = c # a=0, b=1, ..., h=7
        rank_idx = 7 - r # rank 8 = row 0
        sq = chess.square(file_idx, rank_idx)
        
        piece = board.piece_at(sq)
        
        # ✅ ตรวจสอบว่าเป็น Empty
        if piece is None:
            label = "Empty"
        else:
            # Map chess.Piece → label
            color = "W" if piece.color == chess.WHITE else "B"
            ptype = {
                chess.PAWN: "P", chess.ROOK: "R", chess.KNIGHT: "N",
                chess.BISHOP: "B", chess.QUEEN: "Q", chess.KING: "K"
            }[piece.piece_type]
            label = f"{color}{ptype}"
        
        
        # 🌟🌟🌟 FIX: Adaptive Augmentation - บันทึกใน 4 มุมมอง 🌟🌟🌟
        for k in range(4):
            # k=0 (0deg), k=1 (90deg), k=2 (180deg), k=3 (270deg)
            if k == 0:
                rotated_patch = patch
                angle_suffix = ""
            else:
                # Rotates: 90 CW, 180, 270 CW
                rot_code = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
                rotated_patch = cv2.rotate(patch, rot_code[k-1]) 
                angle_suffix = f"_r{k*90}"
            
            # บันทึก
            label_dir = out_path / label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # เพิ่ม suffix (_r0, _r90, _r180, _r270) ในชื่อไฟล์
            fname = f"{Path(video_path).stem}_r{r}c{c}{angle_suffix}.jpg"
            if cv2.imwrite(str(label_dir / fname), rotated_patch):
                saved_count += 1
                
    return saved_count

def simple_augment(img):
    # ฟังก์ชันนี้ไม่ได้ถูกเรียกใช้ในการสร้าง Dataset แต่เราเก็บไว้ตามโครงสร้างเดิม
    warnings.warn(
        "simple_augment() is deprecated and will be removed in a future release. "
        "Use your TF/Keras augmentation pipeline instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), random.uniform(-15,15), 1.0)
    aug = cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)
    alpha = 1.0 + random.uniform(-0.2,0.2) 
    beta  = random.uniform(-20,20) 
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    if random.random() < 0.5:
        k = random.choice([3,5])
        aug = cv2.GaussianBlur(aug, (k,k), 0)
    if random.random() < 0.2:
        x0 = random.randint(0,w-1); y0 = random.randint(0,h-1)
        dx = random.randint(w//10, w//4); dy = random.randint(h//10, h//4)
        cv2.rectangle(aug, (x0,y0), (min(w-1,x0+dx),min(h-1,y0+dy)), (0,0,0), -1)
    return aug