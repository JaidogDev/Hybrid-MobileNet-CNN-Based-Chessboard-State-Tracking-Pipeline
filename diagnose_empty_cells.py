#!/usr/bin/env python3
"""Diagnose Empty cell detection issues"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Paths
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models/cell_cnn.h5"
CLASSES_JSON = ROOT / "models/classes.json"
EMPTY_DIR = ROOT / "data/bootstrap/cells/Empty"
FINAL_TRAIN = ROOT / "data/final/train"

print("="*60)
print("EMPTY CELL DETECTION DIAGNOSTIC")
print("="*60)

# 1. Check class distribution
print("\n1. CLASS DISTRIBUTION IN TRAINING SET:")
print("-"*60)
for class_dir in sorted(FINAL_TRAIN.iterdir()):
    if class_dir.is_dir():
        count = len(list(class_dir.glob("*.jpg")))
        print(f"  {class_dir.name:>5}: {count:4d} images")

# Load model and classes
model = tf.keras.models.load_model(str(MODEL_PATH))
with open(CLASSES_JSON, 'r') as f:
    CLASSES = json.load(f)

print(f"\n2. MODEL PREDICTIONS ON EMPTY CELLS:")
print("-"*60)

# Test on some Empty cells
empty_files = list(EMPTY_DIR.glob("*.jpg"))[:20]  # Test 20 samples
correct = 0
total = len(empty_files)

predictions_summary = {}

for img_path in empty_files:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # Preprocess
    rgb = cv2.cvtColor(cv2.resize(img, (96, 96)), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = preprocess_input(rgb)
    x = np.expand_dims(x, axis=0)

    # Predict
    probs = model.predict(x, verbose=0)[0]
    pred_idx = probs.argmax()
    pred_class = CLASSES[pred_idx]
    pred_conf = float(probs[pred_idx])

    empty_idx = CLASSES.index('Empty')
    empty_conf = float(probs[empty_idx])

    if pred_class == 'Empty':
        correct += 1
        status = "OK"
    else:
        status = "WRONG"

    predictions_summary[pred_class] = predictions_summary.get(pred_class, 0) + 1

    print(f"  {status:5s} {img_path.name[:30]:30s} -> {pred_class:>5s} ({pred_conf:.3f}) | Empty: {empty_conf:.3f}")

print(f"\n3. SUMMARY:")
print("-"*60)
print(f"  Accuracy: {correct}/{total} = {correct/total:.1%}")
print(f"  Predictions breakdown:")
for pred_class, count in sorted(predictions_summary.items(), key=lambda x: -x[1]):
    print(f"    {pred_class:>5s}: {count:2d} times ({count/total:.1%})")

print(f"\n4. RECOMMENDATIONS:")
print("-"*60)
if correct / total < 0.5:
    print("  CRITICAL: Empty cell detection is poor (<50% accuracy)")
    print("  Recommendations:")
    print("    1. Re-extract Empty cells with stricter filtering")
    print("    2. Use class weights during training to balance dataset")
    print("    3. Add more Empty cell augmentation")
    print("    4. Consider using a different model architecture")
elif correct / total < 0.8:
    print("  WARNING: Empty cell detection needs improvement (<80%)")
    print("  Recommendations:")
    print("    1. Add class weights during training")
    print("    2. Increase Empty cell augmentation")
else:
    print("  OK: Empty cell detection is reasonable (>80%)")

print("="*60)
