#!/usr/bin/env python3
"""
Debug script to compare training cells vs inference cells
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
BALANCED_DATA = ROOT / "data/balanced/cells"

# Load sample training images
def load_training_samples():
    """Load samples from training dataset"""
    samples = {}

    for class_name in ["WP", "BP", "BN", "BB", "Empty"]:
        class_dir = BALANCED_DATA / class_name
        if class_dir.exists():
            imgs = list(class_dir.glob("*.jpg"))[:3]  # Get 3 samples
            samples[class_name] = [cv2.imread(str(img)) for img in imgs]

    return samples

# Extract cells from video frame
def extract_inference_cells(video_path):
    """Extract cells from video frame using inference pipeline"""
    from Chess_Detection_Competition.utils import load_config
    from Chess_Detection_Competition.improved_board import warp_board_v2, split_grid_v2
    from Chess_Detection_Competition.board import DEFAULT_CFG as BOARD_DEFAULT_CFG

    cfg = load_config()

    def build_board_cfg(user_board):
        merged = dict(BOARD_DEFAULT_CFG["board"])
        if user_board:
            merged.update(user_board)
        return {"board": merged}

    CFG_FOR_BOARD = build_board_cfg(cfg.get("board", {}))
    IMG_SIZE = int(cfg["cells"]["img_size"])

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    warped, _ = warp_board_v2(frame, CFG_FOR_BOARD, manual_mode=False)
    if warped is None:
        return None

    cells = split_grid_v2(warped, IMG_SIZE)

    # Return dictionary of cells by position
    cell_dict = {}
    for (r, c), patch in cells:
        cell_dict[(r, c)] = patch

    return cell_dict

# Main comparison
def main():
    print("Loading training samples...")
    training_samples = load_training_samples()

    print("Extracting inference cells from video...")
    VIDEOS_DIR = ROOT / "data/public/videos"
    video_files = list(VIDEOS_DIR.glob("*.mp4"))

    if not video_files:
        print("No videos found!")
        return

    inference_cells = extract_inference_cells(video_files[0])

    if inference_cells is None:
        print("Failed to extract cells!")
        return

    # Create comparison figure
    fig, axes = plt.subplots(5, 6, figsize=(15, 12))
    fig.suptitle("Training Samples vs Inference Cells", fontsize=16)

    row = 0
    for class_name in ["WP", "BP", "BN", "BB", "Empty"]:
        # Training samples (columns 0-2)
        if class_name in training_samples:
            for col, img in enumerate(training_samples[class_name][:3]):
                if img is not None:
                    axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    axes[row, col].axis('off')
                    if col == 0:
                        axes[row, col].set_title(f"{class_name}\n(Training)", fontsize=10)
                else:
                    axes[row, col].axis('off')

        # Inference samples (columns 3-5) - find cells with pieces/empty
        inference_col = 3
        for (r, c), patch in list(inference_cells.items())[:20]:  # Check first 20 cells
            if inference_col >= 6:
                break

            # Simple heuristic: show different looking cells
            if inference_col == 3:  # Always show first one
                axes[row, inference_col].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                axes[row, inference_col].axis('off')
                if row == 0:
                    axes[row, inference_col].set_title(f"Inference\nCell ({r},{c})", fontsize=10)
                else:
                    axes[row, inference_col].set_title(f"Cell ({r},{c})", fontsize=8)
                inference_col += 1

        row += 1

    plt.tight_layout()
    plt.savefig("debug/training_vs_inference_cells.jpg", dpi=150, bbox_inches='tight')
    print("Saved: debug/training_vs_inference_cells.jpg")
    plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)

    for class_name, samples in training_samples.items():
        if samples:
            sample = samples[0]
            print(f"\n{class_name}:")
            print(f"  Training sample size: {sample.shape}")
            print(f"  Mean intensity: {sample.mean():.1f}")
            print(f"  Std intensity: {sample.std():.1f}")

    print("\nInference cells:")
    if inference_cells:
        sample_cell = list(inference_cells.values())[0]
        print(f"  Cell size: {sample_cell.shape}")
        print(f"  Mean intensity: {sample_cell.mean():.1f}")
        print(f"  Std intensity: {sample_cell.std():.1f}")

if __name__ == "__main__":
    main()
