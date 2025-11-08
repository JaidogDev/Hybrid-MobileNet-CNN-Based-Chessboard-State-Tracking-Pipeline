# import cv2, numpy as np

# def _order_corners(pts):
#     s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
#     tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
#     tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
#     return np.array([tl,tr,br,bl], dtype=np.float32)

# def _find_board_corners(bgr, cfg):
#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5,5), 0)
#     edges = cv2.Canny(gray, cfg["board"]["canny_low"], cfg["board"]["canny_high"], apertureSize=3)
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, cfg["board"]["hough_threshold"],
#                             minLineLength=cfg["board"]["min_line_length"],
#                             maxLineGap=cfg["board"]["max_line_gap"])
#     if lines is None:
#         raise RuntimeError("No lines found")

#     mask = np.zeros_like(gray)
#     for l in lines:
#         x1,y1,x2,y2 = l[0]
#         cv2.line(mask, (x1,y1), (x2,y2), 255, 2)

#     cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnt = max(cnts, key=cv2.contourArea)
#     epsilon = 0.02*cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     if len(approx) < 4:
#         rect = cv2.minAreaRect(cnt)
#         box = cv2.boxPoints(rect)
#         corners = box.astype(np.float32)
#     else:
#         hull = cv2.convexHull(approx)
#         corners = hull.reshape(-1,2).astype(np.float32)
#         if len(corners) > 4:
#             rect = cv2.minAreaRect(hull)
#             corners = cv2.boxPoints(rect).astype(np.float32)

#     return _order_corners(corners)

# def warp_board(bgr, cfg):
#     w = cfg["board"]["warp_size"]
#     src = _find_board_corners(bgr, cfg)
#     dst = np.float32([[0,0],[w,0],[w,w],[0,w]])
#     M = cv2.getPerspectiveTransform(src, dst)
#     warped = cv2.warpPerspective(bgr, M, (w,w))
#     return warped, M

# def split_grid(warped, cell_px):
#     H, W = warped.shape[:2]
#     assert H==W, "Warp must be square"
#     step = H//8
#     cells = []
#     for r in range(8):
#         for c in range(8):
#             y0,y1 = r*step, (r+1)*step
#             x0,x1 = c*step, (c+1)*step
#             patch = warped[y0:y1, x0:x1]
#             patch = cv2.resize(patch, (cell_px, cell_px), interpolation=cv2.INTER_AREA)
#             cells.append(((r,c), patch))
#     return cells  # [((r,c), BGR)]

# board_detect.py
import cv2
import numpy as np

# ---------------------- config (ปรับได้) ----------------------
DEFAULT_CFG = {
    "board": {
        "warp_size": 800,
        "canny_low": 50,
        "canny_high": 150,
        "hough_threshold": 60,
        "min_line_length": 120,
        "max_line_gap": 10,
        # refine local-search (พิกเซล/องศา/สเกล)
        "refine_dx": 3,
        "refine_dy": 3,
        "refine_rot_deg": 1.0,
        "refine_scale": 0.01,
    }
}

# ---------------------- utils geometry ----------------------
def _order_corners(pts4):
    pts = np.asarray(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _line_abcd(x1,y1,x2,y2):
    # Ax + By + C = 0  (normalized)
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    n = np.hypot(A,B) + 1e-9
    return A/n, B/n, C/n

def _intersect(L1, L2):
    # L: (A,B,C)
    A1,B1,C1 = L1; A2,B2,C2 = L2
    D = A1*B2 - A2*B1
    if abs(D) < 1e-9: 
        return None
    x = (B1*C2 - B2*C1) / D
    y = (C1*A2 - C2*A1) / D
    return np.array([x,y], dtype=np.float32)

def _kmeans_1d(data, k=2, iters=20):
    # simple kmeans สำหรับ 1D angles
    data = data.reshape(-1,1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, iters, 1e-4)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return labels.ravel(), centers.ravel()

# ---------------------- line detection ----------------------
def _detect_lines_edges(bgr, cfg):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    edges = cv2.Canny(gray, cfg["board"]["canny_low"], cfg["board"]["canny_high"])
    return edges

def _fast_lines_or_hough(edges, cfg):
    L = None
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createFastLineDetector"):
        fld = cv2.ximgproc.createFastLineDetector()
        L = fld.detect(edges)  # (N,1,4)
        if L is not None: 
            L = L.reshape(-1,4)
    if L is None or len(L) < 4:
        # fallback HoughLinesP
        hp = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=cfg["board"]["hough_threshold"],
            minLineLength=cfg["board"]["min_line_length"],
            maxLineGap=cfg["board"]["max_line_gap"]
        )
        if hp is None: 
            return None
        L = hp.reshape(-1,4).astype(np.float32)
    return L

def corners_from_extreme_lines(L):
    # 1) angle 0..pi
    dx = L[:,2] - L[:,0]
    dy = L[:,3] - L[:,1]
    ang = (np.arctan2(dy, dx) + np.pi) % np.pi

    # 2) จัด 2 กลุ่ม orientation
    labels, centers = _kmeans_1d(ang, k=2)
    # กลุ่ม0/1
    idx0 = np.where(labels==0)[0]
    idx1 = np.where(labels==1)[0]
    if len(idx0) < 2 or len(idx1) < 2:
        return None

    def extremes(indices):
        # เลือก 2 เส้นที่ "ไกล centroid" ที่สุดด้วย |C| (ระยะจากจุดกำเนิด)
        lines = []
        Cs = []
        for i in indices:
            x1,y1,x2,y2 = L[i]
            A,B,C = _line_abcd(x1,y1,x2,y2)
            lines.append((A,B,C))
            Cs.append(abs(C))
        lines = np.array(lines, dtype=np.float32)
        Cs = np.array(Cs)
        jmax = np.argmax(Cs); jmin = np.argmin(Cs)
        # จริงๆควรเลือกสองด้านซ้าย-ขวา => ใช้ค่าบวกสุด/ลบสุดของ C
        j_pos = np.argmax([l[2] for l in lines])   # C สูงสุด
        j_neg = np.argmin([l[2] for l in lines])   # C ต่ำสุด
        return lines[j_pos], lines[j_neg]

    Lh1, Lh2 = extremes(idx0)
    Lv1, Lv2 = extremes(idx1)

    # 3) จุดตัด 4 มุม (h-v สองคู่)
    p1 = _intersect(Lh1, Lv1)
    p2 = _intersect(Lh1, Lv2)
    p3 = _intersect(Lh2, Lv2)
    p4 = _intersect(Lh2, Lv1)
    P = [p for p in [p1,p2,p3,p4] if p is not None]
    if len(P) != 4: 
        return None
    return _order_corners(np.stack(P, axis=0))

# ---------------------- color mask branch ----------------------
def _mask_board_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m_green = cv2.inRange(hsv, (30, 20, 40), (90, 255, 255))
    m_white = cv2.inRange(hsv, (0, 0, 180), (179, 60, 255))
    mask = cv2.bitwise_or(m_green, m_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    return mask

# ---------------------- checker energy refine ----------------------
def _checker_energy(warped):
    # คิดคะแนนความเป็นลายหมากรุก 8x8
    H,W = warped.shape[:2]
    g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    # สร้าง kernel ลาย chess 8x8 แบบ coarse
    cell = max(4, min(H,W)//16)
    k = np.zeros((cell*8, cell*8), np.float32)
    s = 1.0
    for r in range(8):
        for c in range(8):
            val = 1 if (r+c)%2==0 else -1
            k[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = val
    k = k / np.linalg.norm(k)
    G = cv2.resize(g, (cell*8, cell*8), interpolation=cv2.INTER_AREA)
    G = (G - G.mean()) / (G.std()+1e-6)
    return float(abs((G*k).sum()))

def _perturb_corners(corners, dx, dy, rot_deg, scale):
    # เลื่อน/หมุน/สเกลรอบ centroid
    C = corners.copy().astype(np.float32)
    cen = C.mean(axis=0)
    # translate
    C = C + np.array([dx, dy], np.float32)
    # scale
    C = (C - cen)*(1.0+scale) + cen
    # rotate
    th = np.deg2rad(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]], np.float32)
    C = ((C - cen) @ R.T) + cen
    return C

def refine_by_checker_energy(bgr, rough_corners, grid=8, cfg=DEFAULT_CFG):
    w = cfg["board"]["warp_size"]
    dst = np.float32([[0,0],[w,0],[w,w],[0,w]])
    base = _order_corners(rough_corners)

    best = base.copy()
    best_score = -1e9

    dx_max = cfg["board"]["refine_dx"]
    dy_max = cfg["board"]["refine_dy"]
    rot = cfg["board"]["refine_rot_deg"]
    sca = cfg["board"]["refine_scale"]

    # coarse search ชุดเล็กๆ
    for dx in range(-dx_max, dx_max+1, 2):
        for dy in range(-dy_max, dy_max+1, 2):
            for r in [-rot, 0, rot]:
                for s in [-sca, 0.0, sca]:
                    C = _perturb_corners(base, dx, dy, r, s)
                    M = cv2.getPerspectiveTransform(C.astype(np.float32), dst)
                    warped = cv2.warpPerspective(bgr, M, (w,w))
                    score = _checker_energy(warped)
                    if score > best_score:
                        best_score = score
                        best = C.copy()
    return _order_corners(best)

# ---------------------- main APIs ----------------------
def _find_board_corners(bgr, cfg=DEFAULT_CFG):
    # 0) ตัดขอบดำหนาๆ ออกก่อน (ถ้ามี)
    h,w = bgr.shape[:2]
    pad = int(0.02 * min(h,w))
    bgr_crop = bgr[pad:h-pad, pad:w-pad].copy()
    off = np.array([pad, pad], np.float32)

    # 1) mask สี
    mask = _mask_board_hsv(bgr_crop)
    use_mask = (mask>0).sum() > 0.02 * mask.size

    if use_mask:
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box  = cv2.boxPoints(rect).astype(np.float32)
        rough = _order_corners(box) + off
    else:
        edges = _detect_lines_edges(bgr, cfg)
        L = _fast_lines_or_hough(edges, cfg)
        if L is None or len(L) < 4:
            raise RuntimeError("No lines found")
        corners = corners_from_extreme_lines(L)
        if corners is None:
            raise RuntimeError("Cannot form extreme line polygon")
        rough = corners  # edges ทำบนภาพเต็มอยู่แล้ว

    # 2) refine ด้วย checker energy
    final = refine_by_checker_energy(bgr, rough, grid=8, cfg=cfg)
    return final

def warp_board(bgr, cfg=DEFAULT_CFG):
    w = cfg["board"]["warp_size"]
    src = _find_board_corners(bgr, cfg)
    dst = np.float32([[0,0],[w,0],[w,w],[0,w]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (w,w))
    # return warped, M, src
    return warped, M

def split_grid(warped, cell_px):
    H, W = warped.shape[:2]
    assert H==W, "Warp must be square"
    step = H//8
    cells = []
    for r in range(8):
        for c in range(8):
            y0,y1 = r*step, (r+1)*step
            x0,x1 = c*step, (c+1)*step
            patch = warped[y0:y1, x0:x1]
            patch = cv2.resize(patch, (cell_px, cell_px), interpolation=cv2.INTER_AREA)
            cells.append(((r,c), patch))
    return cells
