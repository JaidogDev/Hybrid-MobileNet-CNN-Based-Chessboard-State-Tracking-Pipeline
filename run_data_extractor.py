# run_data_extractor.py

import os
import json
from pathlib import Path
from glob import glob

# ----------------- 1. เตรียม Path และ Config -----------------

# Path ไปยัง src/Chess_Detection_Competition/
SRC_DIR = Path("src/Chess_Detection_Competition") 

# นำเข้าฟังก์ชัน bootstrap_from_first_frame ที่แก้ไขแล้ว
import src.Chess_Detection_Competition.cells as cell 
# นำเข้า utils เพื่อโหลด config
from src.Chess_Detection_Competition.utils import load_config 

# โหลด config (สมมติว่าคุณมีไฟล์ config/parameters.yaml)
try:
    cfg = load_config("configs/parameters.yaml")
except FileNotFoundError:
    # Fallback ถ้าไม่มีไฟล์ config.yaml
    cfg = {"board": {"warp_size": 640}, "cells": {"img_size": 96}} 
    print("Warning: ใช้ Default Config เนื่องจากไม่พบ parameters.yaml")

# กำหนดโฟลเดอร์สำหรับเก็บภาพหมากที่ดึงออกมา
OUTPUT_DATA_DIR = 'new_training_dataset_from_video'
Path(OUTPUT_DATA_DIR).mkdir(parents=True, exist_ok=True)


# ----------------- 2. ข้อมูลวิดีโอและ FEN (แทนที่ FEN ตรงนี้!) -----------------

VIDEOS_TO_PROCESS = [
    {
        "path": "data/2_Move_rotate_student.mp4",
        "fen": "r2qkbnN/pppb2p1/3p4/n2Pp3/1P2P3/2P5/P1P2KPP/R2Q1B1R b q - 0 1"  # FEN ที่ 1
    },
    {
        "path": "data/2_move_student.mp4",
        "fen": "r2qkbnN/pppb2p1/3p4/n2Pp3/1P2P3/2N5/P1P2KPP/R2Q1B1R b HAhq - 0 1" # FEN ที่ 2 
    },
    {
        "path": "data/4_Move_student.mp4",
        "fen": "r2qkbnr/pppb2p1/3p1p2/n2Pp3/1P2P2P/2N3P1/P1P2PPP/R2QKB1R b KQkq - 0 1" # FEN ที่ 3
    },
    {
        "path": "data/6_Move_student.mp4",
        "fen": "1r2kbnr/pppb2p1/3p4/PN1Pp3/4P3/2P3P1/P1P4P/1R1QKB1R b Kk - 0 1" # FEN ที่ 4
    }
]

# ----------------- 3. การวนลูปเพื่อดึงข้อมูล -----------------

for video_info in VIDEOS_TO_PROCESS:
    video_path = video_info["path"]
    starting_fen = video_info["fen"]
    video_name = Path(video_path).stem

    print(f"\n--- 🎬 กำลังประมวลผลวิดีโอ: {video_name} ---")
    
    try:
        saved_count = cell.bootstrap_from_first_frame(
            video_path=video_path,
            out_dir=OUTPUT_DATA_DIR,
            cfg=cfg,
            start_fen=starting_fen # <-- ส่ง FEN เข้าไป
        )
        print(f"✅ ดึงข้อมูล Cell Patches ได้ {saved_count} ไฟล์ สำหรับ {video_name}")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูลจาก {video_name}: {e}")
        
print("\n--- ✅ เสร็จสิ้นการดึงข้อมูลทั้งหมด ---")

# ตรวจสอบจำนวนไฟล์ทั้งหมด
all_files = glob(f"{OUTPUT_DATA_DIR}/*/*.jpg")
print(f"รวมจำนวนภาพ Cell Patches ที่สร้างได้ทั้งหมด: {len(all_files)} ไฟล์")