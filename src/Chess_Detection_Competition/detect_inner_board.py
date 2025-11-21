"""
Detect INNER 8×8 board (excluding border labels)

Uses fixed margin to crop borders
"""

import cv2
import numpy as np


def crop_to_inner_board(warped: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Crop warped board to inner 8×8 area (no borders)

    Uses fixed 8% margin from each edge

    Args:
        warped: Full warped board
        debug: Save debug images

    Returns:
        Cropped inner board (square)
    """
    H, W = warped.shape[:2]

    # Fixed 8% margin to remove border labels
    margin = int(min(H, W) * 0.08)

    x1 = margin
    y1 = margin
    x2 = W - margin
    y2 = H - margin

    if debug:
        debug_img = warped.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite("debug/inner_board_detected.jpg", debug_img)
        print(f"Cropping to inner board: ({x1}, {y1}) to ({x2}, {y2})")

    # Crop
    inner = warped[y1:y2, x1:x2]

    # Ensure square
    h, w = inner.shape[:2]
    size = min(h, w)
    if h > w:
        start = (h - size) // 2
        inner = inner[start:start+size, :]
    elif w > h:
        start = (w - size) // 2
        inner = inner[:, start:start+size]

    return inner
