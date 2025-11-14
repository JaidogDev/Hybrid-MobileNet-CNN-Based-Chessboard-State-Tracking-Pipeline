# src/chess_tracker/inference.py
from __future__ import annotations
import os, json
from pathlib import Path
from collections import deque
from typing import Tuple, List

import cv2
import numpy as np
import chess
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

from .board import warp_board
from .improved_board import split_grid_v2  # ✅ Use improved version!
from .model import load_model
from .pgn import labels_to_board, san_list_to_pgn


# --------------------------- classes / utils ---------------------------

def _load_classes(root: Path) -> List[str]:
    """Load class order from models/classes.json (fallback to 13 labels)."""
    p = root / "models" / "classes.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # fallback (13 classes)
    return ["Empty","WP","WN","WB","WR","WQ","WK","BP","BN","BB","BR","BQ","BK"]


def _prep_tensor(bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR -> RGB -> preprocess_input -> (1,H,W,3) float32"""
    # ✅ เพิ่ม histogram equalization
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge([l, a, b])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32)
    x   = preprocess_input(rgb)
    return np.expand_dims(x, axis=0)


def rotate_labels(labels: List[List[str]], k: int) -> List[List[str]]:
    x = np.array(labels, dtype=object)
    for _ in range(k):
        x = np.rot90(x, k=-1)
    return x.tolist()


def orientation_score(lbls: List[List[str]]) -> int:
    """ให้คะแนนสูงเมื่อฝั่งล่างเป็นขาวเยอะ ฝั่งบนเป็นดำเยอะ"""
    W = {"WP","WR","WN","WB","WQ","WK"}
    B = {"BP","BR","BN","BB","BQ","BK"}
    
    # ✅ ปรับเพิ่มน้ำหนักให้ตรวจจับ starting position
    bottom_white = sum(lbl in W for r in range(6, 8) for lbl in lbls[r])  # แถว 6-7
    top_black    = sum(lbl in B for r in range(0, 2) for lbl in lbls[r])  # แถว 0-1
    
    # ตรวจจับแถวเบี้ย (pawn row)
    white_pawn_row = sum(lbls[6][c] == "WP" for c in range(8))
    black_pawn_row = sum(lbls[1][c] == "BP" for c in range(8))
    
    return int(bottom_white + top_black + white_pawn_row*2 + black_pawn_row*2)


# --------------------------- core predictor ---------------------------

class TemporalBoardPredictor:
    """
    Per-frame chessboard classifier (8x8) + temporal smoothing.
    - model_path: path to trained Keras model (.h5)
    - img_size : per-cell patch size (must match training, e.g. 96)
    - smooth_k: moving-average window per cell
    """
    def __init__(self, root: Path, model_path: Path, img_size: int = 96, smooth_k: int = 11):
        self.root = Path(root)
        self.model = load_model(str(model_path))
        self.img_size = int(img_size)
        self.CLASSES = _load_classes(self.root)
        self.buffers = [[deque(maxlen=int(smooth_k)) for _ in range(8)] for __ in range(8)]

    def predict_labels8x8(self, warped_bgr: np.ndarray) -> Tuple[List[List[str]], np.ndarray]:
        """Return (labels 8x8, confs 8x8)."""
        cells = split_grid_v2(warped_bgr, self.img_size)  # ✅ Use improved version!
        X = []
        for _, patch in cells:
            X.append(preprocess_input(cv2.cvtColor(cv2.resize(patch, (self.img_size, self.img_size)),
                                                   cv2.COLOR_BGR2RGB).astype(np.float32)))
        X = np.asarray(X, dtype=np.float32)  # (64, H, W, 3)
        probs = self.model.predict(X, verbose=0)  # (64, C)

        labels = [[None]*8 for _ in range(8)]
        confs  = np.zeros((8, 8), dtype=np.float32)
        k = 0
        for r in range(8):
            for c in range(8):
                self.buffers[r][c].append(probs[k])
                avg = np.mean(np.stack(self.buffers[r][c], axis=0), axis=0)
                idx = int(np.argmax(avg))
                labels[r][c] = self.CLASSES[idx]
                confs[r, c]  = float(avg[idx])
                k += 1
        return labels, confs

    def predict_from_frame(self, frame_bgr: np.ndarray, board_cfg: dict) -> Tuple[List[List[str]], np.ndarray, np.ndarray]:
        """Convenience: raw frame -> warp -> labels/conf. Return (labels, confs, warped)."""
        warped, _aux = warp_board(frame_bgr, board_cfg)
        labels, confs = self.predict_labels8x8(warped)
        return labels, confs, warped


# --------------------------- move resolver ---------------------------

def _mask_changes(a: List[List[str]], b: List[List[str]]) -> np.ndarray:
    return np.array([[a[r][c] != b[r][c] for c in range(8)] for r in range(8)], dtype=bool)


def _rc_to_square(r: int, c: int) -> chess.Square:
    file = c
    rank = 7 - r
    return chess.square(file, rank)


def _find_from_to_pair(prev_labels, now_labels, eff_mask):
    srcs, dsts = [], []
    for r in range(8):
        for c in range(8):
            if not eff_mask[r, c]:
                continue
            a, b = prev_labels[r][c], now_labels[r][c]
            if a != "Empty" and b == "Empty":
                srcs.append((r, c, a))
            if a == "Empty" and b != "Empty":
                dsts.append((r, c, b))
    if not srcs or not dsts:
        return None
    # greedy by manhattan dist
    best, best_d = None, 1e9
    for (rs, cs, pa) in srcs:
        for (rd, cd, pb) in dsts:
            d = abs(rs - rd) + abs(cs - cd)
            if d < best_d:
                best_d = d
                best = ((rs, cs, pa), (rd, cd, pb))
    return best


def _make_move_candidates(rs, cs, rd, cd, try_promos=("Q","R","B","N")):
    from_sq = _rc_to_square(rs, cs)
    to_sq   = _rc_to_square(rd, cd)
    cands = [chess.Move(from_sq, to_sq)]
    for sym in try_promos:
        promo = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}[sym]
        cands.append(chess.Move(from_sq, to_sq, promotion=promo))
    return cands


def resolve_move_by_legality(prev_labels, now_labels, eff_mask, try_promos=("Q","R","B","N")):
    b_prev = labels_to_board(prev_labels)
    pair = _find_from_to_pair(prev_labels, now_labels, eff_mask)
    if pair is None:
        return None
    (rs, cs, _), (rd, cd, _) = pair
    candidates = _make_move_candidates(rs, cs, rd, cd, try_promos=try_promos)
    legal = set(b_prev.legal_moves)
    for mv in candidates:
        if mv in legal:
            return mv
    return None


# --------------------------- high-level video runner ---------------------------

def decode_video_to_pgn(
    video_path: Path,
    predictor: TemporalBoardPredictor,
    board_cfg: dict,
    *,
    tau: float = 0.60,
    sample_step: int = 1,
    require_stable_frames: int = 3,
    min_changes: int = 2,
    max_changes: int = 6,
    enforce_legal: bool = True,
    pending_horizon: int = 4,
    warmup_steps: int = 3,
) -> str:
    """
    Process a whole video -> PGN string.
    - Uses single homography per frame (no H0-scan for simplicity).
    - Orientation is chosen once from the first frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        return ""

    # frame0
    warped0, _ = warp_board(frame0, board_cfg)
    if warped0 is None or getattr(warped0, "size", 0) == 0:
        cap.release()
        return ""

    # predict first
    labels0, confs0 = predictor.predict_labels8x8(warped0)
    # orientation (pick best k)
    cands = [(k, rotate_labels(labels0, k)) for k in range(4)]
    ORIENT_K, prev_labels = max(cands, key=lambda t: orientation_score(t[1]))
    prev_confs = confs0.copy()

    # init stability
    buffers_len = predictor.buffers[0][0].maxlen
    stable = [row[:] for row in prev_labels]
    steady = [[2]*8 for _ in range(8)]  # already stable-ish

    # warm-up: fill buffers
    for _ in range(warmup_steps):
        ok = cap.grab()
        if not ok: break
        ok, fr = cap.retrieve()
        if not ok or fr is None: break
        warped, _ = warp_board(fr, board_cfg)
        if warped is None or getattr(warped, "size", 0) == 0:
            continue
        now_labels_raw, _ = predictor.predict_labels8x8(warped)
        _ = rotate_labels(now_labels_raw, ORIENT_K)  # just to fill buffers

    # run
    sans = []
    frame_id = warmup_steps
    pending = None

    while True:
        ok = cap.read()
        if not ok[0]:
            break
        frame = ok[1]
        if frame is None:
            break
        frame_id += 1
        if (frame_id % max(1, sample_step)) != 0:
            continue

        warped, _ = warp_board(frame, board_cfg)
        if warped is None or getattr(warped, "size", 0) == 0:
            continue

        now_labels_raw, now_confs = predictor.predict_labels8x8(warped)
        # orient
        for _ in range(ORIENT_K):
            now_labels_raw = rotate_labels(now_labels_raw, 1)
            now_confs = np.rot90(now_confs, k=-1)

        # sticky by tau
        sticky = [[now_labels_raw[r][c] if now_confs[r, c] >= tau else stable[r][c]
                   for c in range(8)] for r in range(8)]

        # per-cell stability
        for r in range(8):
            for c in range(8):
                if sticky[r][c] == stable[r][c]:
                    steady[r][c] = min(steady[r][c] + 1, require_stable_frames)
                else:
                    steady[r][c] = 1
                if steady[r][c] >= require_stable_frames:
                    stable[r][c] = sticky[r][c]

        # diff + gating
        conf_or  = (prev_confs >= tau) | (now_confs >= tau)
        diff_raw = _mask_changes(stable, sticky)
        eff_mask = conf_or & diff_raw
        n_changes = int(eff_mask.sum())

        committed = False
        if min_changes <= n_changes <= max_changes:
            mv = None
            if enforce_legal:
                mv = resolve_move_by_legality(stable, sticky, eff_mask)
            if mv is None:
                # naive (no legality) fallback
                b_prev = labels_to_board(stable)
                mv = None
                # simple 2-square diff
                pair = _find_from_to_pair(stable, sticky, eff_mask)
                if pair:
                    (rs, cs, _), (rd, cd, _) = pair
                    mv = chess.Move(_rc_to_square(rs, cs), _rc_to_square(rd, cd))
                    if enforce_legal and mv not in set(b_prev.legal_moves):
                        mv = None
            if mv is not None:
                b_prev = labels_to_board(stable)
                try:
                    san = b_prev.san(mv)
                except Exception:
                    san = None
                if san:
                    sans.append(san)
                    prev_labels = [row[:] for row in sticky]
                    prev_confs  = now_confs.copy()
                    stable      = [row[:] for row in sticky]
                    steady      = [[require_stable_frames]*8 for _ in range(8)]
                    pending     = None
                    committed   = True
            else:
                pending = {"frame": frame_id, "mask": eff_mask.copy()}

        # pending combine (optional simple)
        if (not committed) and (pending is not None) and ((frame_id - pending["frame"]) <= pending_horizon):
            combined_mask = pending["mask"] | eff_mask
            mv2 = resolve_move_by_legality(stable, sticky, combined_mask)
            if mv2 is not None:
                b_prev = labels_to_board(stable)
                try:
                    san = b_prev.san(mv2)
                except Exception:
                    san = None
                if san:
                    sans.append(san)
                    prev_labels = [row[:] for row in sticky]
                    prev_confs  = now_confs.copy()
                    stable      = [row[:] for row in sticky]
                    steady      = [[require_stable_frames]*8 for _ in range(8)]
                    pending     = None

        if pending is not None and (frame_id - pending["frame"]) > pending_horizon:
            pending = None

    cap.release()
    try:
        return san_list_to_pgn(sans)
    except Exception:
        return " ".join(sans)

def preview_video_with_overlay(
    video_path, out_path, predictor, cfg_for_board,
    tau=0.6, sample_step=1,
    require_stable_frames=3,
    min_changes=2, max_changes=6,
    enforce_legal=True,
    pending_horizon=4,
    warmup_steps=3
):
    """
    วาด annotation ลงบน video (overlay 8x8 grid + label + conf)
    และ export ออกเป็นไฟล์ mp4 พร้อมแสดง PGN สรุป
    """
    import cv2, numpy as np
    from .board import warp_board
    from .pgn import diff_to_move, san_list_to_pgn

    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    prev_labels = None
    san_list = []

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % sample_step != 0:
            i += 1
            continue

        warped, _ = warp_board(frame, cfg_for_board)
        labels, confs = predictor.predict_labels8x8(warped)

        # วาดกรอบ
        H, W = warped.shape[:2]
        xs = np.linspace(0, W, 9).astype(int)
        ys = np.linspace(0, H, 9).astype(int)
        for x in xs: cv2.line(warped, (x,0), (x,H-1), (255,255,255), 1, cv2.LINE_AA)
        for y in ys: cv2.line(warped, (0,y), (W-1,y), (255,255,255), 1, cv2.LINE_AA)
        for r in range(8):
            for c in range(8):
                if confs[r,c] < tau:
                    continue
                text = labels[r][c]
                cv2.putText(warped, text, (xs[c]+6, ys[r]+24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # detect move diff
        if prev_labels is not None:
            move = diff_to_move(prev_labels, labels)
            if move:
                san_list.append(move)
                print(f"[frame {i}] move={move}")
        prev_labels = labels

        # init writer
        if out is None:
            out = cv2.VideoWriter(str(out_path), fourcc, 15, (W,H))
        out.write(warped)
        i += 1

    cap.release()
    if out: out.release()
    pgn = san_list_to_pgn(san_list)
    return pgn
