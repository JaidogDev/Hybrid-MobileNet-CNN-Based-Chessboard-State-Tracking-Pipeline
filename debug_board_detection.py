#!/usr/bin/env python3
"""
Debug script to visualize board detection steps
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def debug_board_detection(video_path, cfg):
    """Show all steps of board detection"""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("Cannot read video")
        return

    print(f"Frame size: {frame.shape}")

    # Show original
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Frame")
    axes[0, 0].axis('off')

    # HSV mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 20, 80])
    upper = np.array([25, 100, 220])
    mask = cv2.inRange(hsv, lower, upper)

    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title("2. HSV Mask (Light Wooden)")
    axes[0, 1].axis('off')

    # Edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 50, 150)

    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title("3. Canny Edges")
    axes[0, 2].axis('off')

    # Detect corners
    from Chess_Detection_Competition.improved_board import warp_board_v2, _auto_detect_corners

    corners = _auto_detect_corners(frame)

    # Draw detected corners
    frame_with_corners = frame.copy()
    if corners is not None:
        for i, (x, y) in enumerate(corners):
            cv2.circle(frame_with_corners, (int(x), int(y)), 10, (0, 255, 0), -1)
            cv2.putText(frame_with_corners, str(i+1), (int(x)+15, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw lines between corners
        pts = corners.astype(int)
        cv2.polylines(frame_with_corners, [pts], True, (0, 255, 0), 2)

    axes[1, 0].imshow(cv2.cvtColor(frame_with_corners, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("4. Detected Corners")
    axes[1, 0].axis('off')

    # Warp
    if corners is not None:
        warped, _ = warp_board_v2(frame, cfg, manual_mode=False)

        if warped is not None:
            axes[1, 1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title("5. Warped Board")
            axes[1, 1].axis('off')

            # Draw grid on warped
            H, W = warped.shape[:2]
            margin = int(0.05 * min(H, W))
            xs = np.linspace(margin, W - margin, 9).astype(int)
            ys = np.linspace(margin, H - margin, 9).astype(int)

            grid = warped.copy()
            for x in xs:
                cv2.line(grid, (x, 0), (x, H-1), (0, 255, 0), 2)
            for y in ys:
                cv2.line(grid, (0, y), (W-1, y), (255, 0, 0), 2)

            axes[1, 2].imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title("6. Grid (5% margin)")
            axes[1, 2].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, "Warp failed", ha='center', va='center')
            axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, "No corners detected", ha='center', va='center')
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('debug/board_detection_steps.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: debug/board_detection_steps.jpg")

if __name__ == "__main__":
    from Chess_Detection_Competition.utils import load_config
    from Chess_Detection_Competition.board import DEFAULT_CFG as BOARD_DEFAULT_CFG

    cfg = load_config()
    ROOT = Path(__file__).parent
    VIDEOS_DIR = ROOT / cfg["paths"]["videos_dir"]

    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    if video_files:
        print(f"Debugging: {video_files[0].name}")

        def build_board_cfg(user_board):
            merged = dict(BOARD_DEFAULT_CFG["board"])
            if user_board:
                merged.update(user_board)
            return {"board": merged}

        CFG_FOR_BOARD = build_board_cfg(cfg.get("board", {}))
        debug_board_detection(video_files[0], CFG_FOR_BOARD)
    else:
        print("No videos found")
