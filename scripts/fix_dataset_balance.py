#!/usr/bin/env python3
"""
Fix class imbalance by generating more Empty cells and balancing the dataset.
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path
import shutil
from tqdm import tqdm

def augment_empty_cells(src_dir, target_count=2000):
    """Generate more empty cells through augmentation"""
    src_path = Path(src_dir) / "Empty"
    if not src_path.exists():
        print(f"❌ No Empty folder found at {src_path}")
        return 0
        
    existing_files = list(src_path.glob("*.jpg"))
    if not existing_files:
        print(f"❌ No images found in {src_path}")
        return 0
    
    print(f"Found {len(existing_files)} existing Empty cells")
    print(f"Target: {target_count} Empty cells")
    
    if len(existing_files) >= target_count:
        print("Already have enough Empty cells")
        return len(existing_files)
    
    needed = target_count - len(existing_files)
    print(f"Need to generate {needed} more Empty cells")
    
    generated = 0
    base_name = "aug_empty"
    
    for i in tqdm(range(needed), desc="Generating Empty cells"):
        # Pick random source image
        src_file = random.choice(existing_files)
        img = cv2.imread(str(src_file))
        
        if img is None:
            continue
            
        # Apply random augmentation
        aug_img = augment_image(img)
        
        # Save augmented image
        out_file = src_path / f"{base_name}_{i:06d}.jpg"
        if cv2.imwrite(str(out_file), aug_img):
            generated += 1
    
    print(f"Generated {generated} new Empty cells")
    return len(existing_files) + generated

def augment_image(img):
    """Apply random augmentations to an image"""
    h, w = img.shape[:2]
    
    # Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Random brightness/contrast
    alpha = random.uniform(0.8, 1.2)  # Contrast
    beta = random.uniform(-20, 20)    # Brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Random noise
    if random.random() < 0.3:
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Random blur
    if random.random() < 0.2:
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Random horizontal flip (50% chance)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    
    return img

def balance_all_classes(bootstrap_dir, public_dir, output_dir, target_per_class=1000):
    """Balance all classes to have similar number of samples"""
    
    bootstrap_path = Path(bootstrap_dir)
    public_path = Path(public_dir) 
    output_path = Path(output_dir)
    
    # Classes to balance
    classes = ["Empty", "WP", "WN", "WB", "WR", "WQ", "WK", "BP", "BN", "BB", "BR", "BQ", "BK"]
    
    print("Current class distribution:")
    print("=" * 50)
    
    total_moved = 0
    
    for cls in classes:
        # Count existing samples
        bootstrap_cls = bootstrap_path / cls
        public_cls = public_path / cls
        output_cls = output_path / cls
        
        bootstrap_count = len(list(bootstrap_cls.glob("*.jpg"))) if bootstrap_cls.exists() else 0
        public_count = len(list(public_cls.glob("*.jpg"))) if public_cls.exists() else 0
        total_count = bootstrap_count + public_count
        
        print(f"{cls:>8}: Bootstrap={bootstrap_count:>4}, Public={public_count:>4}, Total={total_count:>4}")
        
        # Create output directory
        output_cls.mkdir(parents=True, exist_ok=True)
        
        # Copy all existing files first
        moved = 0
        if bootstrap_cls.exists():
            for f in bootstrap_cls.glob("*.jpg"):
                shutil.copy2(f, output_cls / f.name)
                moved += 1
        
        if public_cls.exists():
            for f in public_cls.glob("*.jpg"):
                shutil.copy2(f, output_cls / f.name)
                moved += 1
        
        total_moved += moved
        
        # Augment if needed (especially for Empty class)
        if cls == "Empty" and total_count < target_per_class:
            existing_files = list(output_cls.glob("*.jpg"))
            if existing_files:
                needed = target_per_class - len(existing_files)
                print(f"Augmenting {cls}: need {needed} more samples")
                
                for i in range(needed):
                    src_file = random.choice(existing_files)
                    img = cv2.imread(str(src_file))
                    if img is not None:
                        aug_img = augment_image(img)
                        out_file = output_cls / f"aug_{cls}_{i:06d}.jpg"
                        cv2.imwrite(str(out_file), aug_img)
    
    print("=" * 50)
    print(f"Balanced dataset created in {output_path}")
    print(f"Total files processed: {total_moved}")

if __name__ == "__main__":
    # Paths
    ROOT = Path(".").resolve()
    BOOTSTRAP_DIR = ROOT / "data/bootstrap/cells"
    PUBLIC_DIR = ROOT / "data/public/cells" 
    BALANCED_DIR = ROOT / "data/balanced/cells"
    
    print("Starting dataset balancing...")
    print(f"Bootstrap: {BOOTSTRAP_DIR}")
    print(f"Public: {PUBLIC_DIR}")
    print(f"Output: {BALANCED_DIR}")
    
    # First, balance all classes
    balance_all_classes(BOOTSTRAP_DIR, PUBLIC_DIR, BALANCED_DIR, target_per_class=800)
    
    # Then augment Empty specifically if still needed
    empty_count = len(list((BALANCED_DIR / "Empty").glob("*.jpg")))
    if empty_count < 1500:
        print(f"\nEmpty class still needs more samples: {empty_count}/1500")
        augment_empty_cells(BALANCED_DIR, target_count=1500)
    
    print("\nDataset balancing complete!")