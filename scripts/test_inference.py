#!/usr/bin/env python3
"""
Quick inference test with the new lightweight model
"""

import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

def test_new_model_on_video():
    """Test the new lightweight model on a video"""
    
    ROOT = Path(".").resolve()
    
    # Load new model
    model_path = ROOT / "models/cell_cnn_lightweight.h5"
    classes_path = ROOT / "models/classes.json"
    
    if not model_path.exists():
        print("❌ Model not found. Train first:")
        print("python scripts/lightweight_training.py")
        return
    
    print("📂 Loading lightweight model...")
    model = tf.keras.models.load_model(model_path)
    
    with open(classes_path, 'r') as f:
        class_names = json.load(f)
    
    print(f"✅ Model loaded with {len(class_names)} classes")
    print(f"📊 Classes: {class_names}")
    
    # Load video
    from Chess_Detection_Competition.utils import load_config
    cfg = load_config()
    videos_dir = ROOT / cfg['paths']['videos_dir']
    videos = list(videos_dir.glob("*.mp4"))
    
    if not videos:
        print("❌ No videos found")
        return
    
    video_path = videos[0]
    print(f"🎬 Testing on: {video_path.name}")
    
    # Process first frame
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    
    if not ok:
        print("❌ Cannot read video frame")
        return
    
    # Quick board detection (use improved version)
    from Chess_Detection_Competition.improved_board import warp_board_v2
    
    board_cfg = {'board': cfg.get('board', {})}
    warped, _ = warp_board_v2(frame, board_cfg, manual_mode=False)
    
    if warped is None:
        print("❌ Board detection failed")
        return
    
    print(f"✅ Board detected: {warped.shape}")
    
    # Extract 8x8 cells
    H, W = warped.shape[:2]
    cell_size = 96
    
    # Simple grid extraction
    cells = []
    for r in range(8):
        for c in range(8):
            y1 = int(r * H / 8)
            y2 = int((r + 1) * H / 8)
            x1 = int(c * W / 8)
            x2 = int((c + 1) * W / 8)
            
            cell = warped[y1:y2, x1:x2]
            cell = cv2.resize(cell, (cell_size, cell_size))
            cells.append(cell)
    
    # Predict all cells
    cells_array = np.array(cells) / 255.0  # Normalize
    predictions = model.predict(cells_array, verbose=0)
    
    # Convert predictions to labels
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Display results
    print("\n🏁 Chess Board Prediction Results:")
    print("=" * 50)
    
    board_2d = []
    for r in range(8):
        row = []
        for c in range(8):
            idx = r * 8 + c
            class_idx = predicted_classes[idx]
            conf = confidences[idx]
            class_name = class_names[class_idx]
            row.append(f"{class_name}")
        board_2d.append(row)
    
    # Print board
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    print("    " + "  ".join([f"{f:>8}" for f in files]))
    for r in range(8):
        rank = 8 - r
        row_str = f"{rank}   "
        for c in range(8):
            idx = r * 8 + c
            class_name = class_names[predicted_classes[idx]]
            conf = confidences[idx]
            if conf >= 0.5:  # High confidence
                row_str += f"{class_name:>8}  "
            else:  # Low confidence
                row_str += f"({class_name:>6})  "
        print(row_str)
    
    # Statistics
    high_conf_count = np.sum(confidences >= 0.5)
    empty_count = np.sum([class_names[c] == "Empty" for c in predicted_classes])
    
    print("=" * 50)
    print(f"📊 Statistics:")
    print(f"   High confidence (≥0.5): {high_conf_count}/64 ({high_conf_count/64*100:.1f}%)")
    print(f"   Empty cells detected: {empty_count}")
    print(f"   Average confidence: {confidences.mean():.3f}")
    
    # Check if results look reasonable for a chess game
    piece_counts = {}
    for class_idx in predicted_classes:
        class_name = class_names[class_idx]
        piece_counts[class_name] = piece_counts.get(class_name, 0) + 1
    
    print(f"\n♟️ Piece Distribution:")
    for piece, count in sorted(piece_counts.items()):
        print(f"   {piece:>6}: {count:>2}")
    
    # Sanity check
    reasonable = True
    issues = []
    
    if empty_count < 10:
        issues.append("Too few empty cells")
        reasonable = False
    
    if high_conf_count < 32:  # At least half should be confident
        issues.append("Low confidence overall")
        reasonable = False
    
    if reasonable:
        print(f"\n✅ Results look reasonable! Model is working.")
    else:
        print(f"\n⚠️ Issues detected: {', '.join(issues)}")
        print("Consider more training or data balancing.")
    
    return reasonable

if __name__ == "__main__":
    print("🧪 Testing New Lightweight Model")
    print("=" * 40)
    
    success = test_new_model_on_video()
    
    if success:
        print("\n🎉 Model test PASSED!")
        print("Ready for Kaggle submission!")
    else:
        print("\n🔧 Model needs improvement.")
        print("Try retraining with more epochs.")