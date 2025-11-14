#!/usr/bin/env python3
"""Quick dataset creation without TensorFlow imports"""

import os
import shutil
import random
from pathlib import Path

def main():
    # Paths
    CELLS_PUBLIC = Path('data/public/cells')
    CELLS_BOOTSTRAP = Path('data/bootstrap/cells') 
    FINAL_TRAIN = Path('data/final/train')
    FINAL_VAL = Path('data/final/val')
    CLASSES_13 = ['Empty','WP','WN','WB','WR','WQ','WK','BP','BN','BB','BR','BQ','BK']

    print("Starting dataset creation...")
    
    # Clean and create directories
    for p in [FINAL_TRAIN, FINAL_VAL]:
        if p.exists():
            shutil.rmtree(p)
        for c in CLASSES_13:
            (p / c).mkdir(parents=True, exist_ok=True)

    # Merge and split
    rng = random.Random(2025)
    
    for c in CLASSES_13:
        print(f"Processing class: {c}")
        
        # Collect all files for this class
        pool = []
        for src_dir in [CELLS_PUBLIC, CELLS_BOOTSTRAP]:
            class_dir = src_dir / c
            if class_dir.exists():
                files = [f for f in class_dir.glob('*.jpg')]
                pool.extend([str(f) for f in files])
        
        if not pool:
            print(f"  No files found for {c}")
            continue
            
        print(f"  Found {len(pool)} files")
        
        # Shuffle and split
        rng.shuffle(pool)
        n_val = max(1, int(len(pool) * 0.1))
        val_set = pool[:n_val]
        train_set = pool[n_val:]
        
        # Copy files
        for file_path in train_set:
            basename = os.path.basename(file_path)
            dest = FINAL_TRAIN / c / basename
            shutil.copy2(file_path, dest)
            
        for file_path in val_set:
            basename = os.path.basename(file_path)
            dest = FINAL_VAL / c / basename
            shutil.copy2(file_path, dest)
        
        print(f"  Copied: train={len(train_set)} val={len(val_set)}")

    print("\nFinal counts:")
    for c in CLASSES_13:
        train_count = len(list((FINAL_TRAIN / c).glob('*')))
        val_count = len(list((FINAL_VAL / c).glob('*')))
        print(f"{c:>5}: train={train_count:4d} val={val_count:4d} total={train_count + val_count:4d}")
    
    print("Dataset creation complete!")

if __name__ == "__main__":
    main()