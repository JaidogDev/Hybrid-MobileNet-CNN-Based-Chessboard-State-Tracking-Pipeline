# run_debug_warp.py

import cv2
from pathlib import Path
import numpy as np
from src.Chess_Detection_Competition.improved_board import warp_board_v2
from src.Chess_Detection_Competition.utils import load_config 

# --- CONFIGURATION ---
VIDEO_PATH = "data/2_move_student.mp4" 
OUTPUT_DIR = "debug"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

try:
    cfg = load_config("configs/parameters.yaml")
except FileNotFoundError:
    cfg = {"board": {"warp_size": 640}, "cells": {"img_size": 96}} 
    print("Warning: ใช้ Default Config เนื่องจากไม่พบ parameters.yaml")


cap = cv2.VideoCapture(VIDEO_PATH)
ok, frame = cap.read()
cap.release()

if not ok or frame is None:
    print(f"❌ ERROR: Cannot read video file: {VIDEO_PATH}")
    exit()

print(f"--- Running Board Detection on {VIDEO_PATH} ---")

try:
    # 1. รัน Board Detection และ Warp
    warped, M = warp_board_v2(frame, cfg)

    # 2. บันทึก Warped Board
    cv2.imwrite(f"{OUTPUT_DIR}/1_warped_board.jpg", warped)
    print(f"✅ Saved 1_warped_board.jpg")

    # 3. วาด Grid Overlay (FIX: เพิ่ม Margin 8% ในการคำนวณ Grid)
    H, W = warped.shape[:2]
    margin = int(min(H, W) * 0.08)

    # Grid เริ่มจาก Margin ถึง (W - Margin)
    xs = np.linspace(margin, W - margin, 9)
    ys = np.linspace(margin, H - margin, 9)
    
    vis = warped.copy()
    for x in xs:
        cv2.line(vis, (int(x), 0), (int(x), H), (0, 255, 0), 2)
    for y in ys:
        cv2.line(vis, (0, int(y)), (W, int(y)), (255, 0, 0), 2)
        
    cv2.imwrite(f"{OUTPUT_DIR}/2_warped_grid.jpg", vis)
    print(f"✅ Saved 2_warped_grid.jpg (Grid Overlay)")

except Exception as e:
    print(f"❌ ERROR in Warping/Debug: {e}")