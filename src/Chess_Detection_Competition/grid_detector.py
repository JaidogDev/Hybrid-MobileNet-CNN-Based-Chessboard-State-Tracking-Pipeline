"""
Grid Detector - Trivial Split Method

Detects inner 8×8 board (no borders), then simple division
"""

import cv2
import numpy as np
from typing import List, Tuple
from .detect_inner_board import crop_to_inner_board


def detect_and_split_grid(warped: np.ndarray, cell_px: int = 96, debug: bool = False) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    """
    Split warped board into 8×8 cells

    1. Detect INNER board (exclude borders)
    2. Simple 8×8 division

    Args:
        warped: Warped board image (may include borders)
        cell_px: Target cell size for output
        debug: If True, save debug images

    Returns:
        List of ((row, col), cell_image) tuples
    """
    # First, crop to inner 8×8 board (no borders)
    inner_board = crop_to_inner_board(warped, debug=debug)

    H, W = inner_board.shape[:2]

    # Simple 8×8 division on the INNER board
    h_lines = [int(y) for y in np.linspace(0, H, 9)]
    v_lines = [int(x) for x in np.linspace(0, W, 9)]

    if debug:
        grid_img = inner_board.copy()
        for y in h_lines:
            cv2.line(grid_img, (0, y), (W, y), (0, 255, 0), 2)
        for x in v_lines:
            cv2.line(grid_img, (x, 0), (x, H), (0, 255, 0), 2)
        cv2.imwrite("debug/detected_grid.jpg", grid_img)
        print("Saved debug/detected_grid.jpg")

    # Extract cells
    cells = []
    for r in range(8):
        y1 = h_lines[r]
        y2 = h_lines[r + 1]

        for c in range(8):
            x1 = v_lines[c]
            x2 = v_lines[c + 1]

            cell = inner_board[y1:y2, x1:x2]

            if cell.shape[0] > 0 and cell.shape[1] > 0:
                cell_resized = cv2.resize(cell, (cell_px, cell_px))
                cells.append(((r, c), cell_resized))

    return cells
