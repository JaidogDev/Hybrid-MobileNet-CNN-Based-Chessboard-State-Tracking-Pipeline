"""
Chess Detection Pipeline
Outputs: 1) Annotated images, 2) CSV, 3) Video with predictions
"""

import cv2
import numpy as np
import json
import csv
import os
from pathlib import Path
from tensorflow.keras.models import load_model
from src.Chess_Detection_Competition.improved_board import warp_board_v2
from src.Chess_Detection_Competition.grid_detector import detect_and_split_grid

# ===== CONFIGURATION =====
VIDEO_PATH = "data/public/videos/2_move_student.mp4"
MODEL_PATH = "models/finetuned_model.h5"
CLASSES_PATH = "models/classes_finetuned.json"

# Output settings
OUTPUT_DIR = "output"
SAVE_IMAGES_EVERY = 30  # Save annotated image every N frames (0 = no images)
PROCESS_EVERY = 1       # Process every N frames (1 = all frames)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

# ===== LOAD MODEL =====
print("Loading model and classes...")
model = load_model(MODEL_PATH)
with open(CLASSES_PATH, 'r') as f:
    class_mapping = json.load(f)
print(f"Loaded {len(class_mapping)} classes")

# ===== HELPER FUNCTIONS =====
def board_to_fen(board_state):
    """Convert 8x8 board state to FEN notation"""
    fen_rows = []
    for r in range(8):
        fen_row = ""
        empty_count = 0
        for c in range(8):
            piece = board_state[r][c]
            if piece == "Empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                # Convert piece names to FEN notation
                fen_map = {
                    "WP": "P", "WN": "N", "WB": "B", "WR": "R", "WQ": "Q", "WK": "K",
                    "BP": "p", "BN": "n", "BB": "b", "BR": "r", "BQ": "q", "BK": "k"
                }
                fen_row += fen_map.get(piece, "?")
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + " w - - 0 1"  # Basic FEN (assumes white to move)

def draw_predictions(frame, board_state, M):
    """Draw predictions on the original frame"""
    overlay = frame.copy()

    # Draw grid lines (8x8)
    margin = int(640 * 0.08)  # 8% margin
    inner_size = 640 - 2 * margin
    h_lines = [int(y) for y in np.linspace(0, inner_size, 9)]
    v_lines = [int(x) for x in np.linspace(0, inner_size, 9)]

    # Transform grid lines to original image
    for h_line in h_lines:
        y_warped = margin + h_line
        pt1_warped = np.array([[margin, y_warped]], dtype=np.float32)
        pt2_warped = np.array([[640 - margin, y_warped]], dtype=np.float32)
        pt1 = cv2.perspectiveTransform(pt1_warped.reshape(-1, 1, 2), np.linalg.inv(M))
        pt2 = cv2.perspectiveTransform(pt2_warped.reshape(-1, 1, 2), np.linalg.inv(M))
        cv2.line(overlay, tuple(pt1.reshape(-1, 2)[0].astype(int)),
                tuple(pt2.reshape(-1, 2)[0].astype(int)), (0, 255, 0), 2)

    for v_line in v_lines:
        x_warped = margin + v_line
        pt1_warped = np.array([[x_warped, margin]], dtype=np.float32)
        pt2_warped = np.array([[x_warped, 640 - margin]], dtype=np.float32)
        pt1 = cv2.perspectiveTransform(pt1_warped.reshape(-1, 1, 2), np.linalg.inv(M))
        pt2 = cv2.perspectiveTransform(pt2_warped.reshape(-1, 1, 2), np.linalg.inv(M))
        cv2.line(overlay, tuple(pt1.reshape(-1, 2)[0].astype(int)),
                tuple(pt2.reshape(-1, 2)[0].astype(int)), (0, 255, 0), 2)

    # Draw predictions text
    for r in range(8):
        for c in range(8):
            piece = board_state[r][c]
            if piece != "Empty":
                # Calculate cell center
                row_start = h_lines[r]
                row_end = h_lines[r + 1]
                col_start = v_lines[c]
                col_end = v_lines[c + 1]
                center_y = (row_start + row_end) // 2 + margin
                center_x = (col_start + col_end) // 2 + margin

                # Transform to original image
                center_warped = np.array([[center_x, center_y]], dtype=np.float32)
                center_orig = cv2.perspectiveTransform(center_warped.reshape(-1, 1, 2), np.linalg.inv(M))
                center = tuple(center_orig.reshape(-1, 2)[0].astype(int))

                # Draw piece name
                cv2.putText(overlay, piece, (center[0] - 15, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return overlay

# ===== PROCESS VIDEO =====
print(f"\nProcessing video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Total frames: {total_frames}, FPS: {fps}, Size: {width}x{height}")

# Setup CSV output
csv_path = f"{OUTPUT_DIR}/predictions.csv"
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'fen'])

# Setup video output
video_path = f"{OUTPUT_DIR}/annotated_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

frame_idx = 0
processed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process every Nth frame
    if frame_idx % PROCESS_EVERY != 0:
        video_writer.write(frame)  # Write original frame
        frame_idx += 1
        continue

    try:
        # Detect board
        warped, M = warp_board_v2(frame, {"board": {"warp_size": 640}})

        # Extract cells
        cells = detect_and_split_grid(warped, cell_px=96)

        # Predict each cell
        board_state = [['' for _ in range(8)] for _ in range(8)]
        for (r, c), cell_img in cells:
            cell_batch = np.expand_dims(cell_img, axis=0) / 255.0
            pred = model.predict(cell_batch, verbose=0)
            class_idx = np.argmax(pred[0])
            class_name = class_mapping[str(class_idx)]
            board_state[r][c] = class_name

        # Convert to FEN
        fen = board_to_fen(board_state)
        csv_writer.writerow([frame_idx, fen])

        # Draw predictions on frame
        annotated_frame = draw_predictions(frame, board_state, M)
        video_writer.write(annotated_frame)

        # Save image every N frames
        if SAVE_IMAGES_EVERY > 0 and processed_count % SAVE_IMAGES_EVERY == 0:
            img_path = f"{OUTPUT_DIR}/images/frame_{frame_idx:06d}.jpg"
            cv2.imwrite(img_path, annotated_frame)
            print(f"Saved image: {img_path}")

        processed_count += 1
        if processed_count % 30 == 0:
            print(f"Processed {processed_count} frames ({frame_idx}/{total_frames})")

    except Exception as e:
        print(f"Error at frame {frame_idx}: {e}")
        video_writer.write(frame)  # Write original frame on error

    frame_idx += 1

# Cleanup
cap.release()
video_writer.release()
csv_file.close()

print(f"\n===== COMPLETE =====")
print(f"Processed {processed_count} frames")
print(f"Outputs:")
print(f"  1. CSV: {csv_path}")
print(f"  2. Video: {video_path}")
print(f"  3. Images: {OUTPUT_DIR}/images/")
