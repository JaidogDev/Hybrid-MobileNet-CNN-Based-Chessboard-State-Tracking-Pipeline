import cv2, numpy as np

def _order_corners(pts):
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

def _find_board_corners(bgr, cfg):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, cfg["board"]["canny_low"], cfg["board"]["canny_high"], apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, cfg["board"]["hough_threshold"],
                            minLineLength=cfg["board"]["min_line_length"],
                            maxLineGap=cfg["board"]["max_line_gap"])
    if lines is None:
        raise RuntimeError("No lines found")

    mask = np.zeros_like(gray)
    for l in lines:
        x1,y1,x2,y2 = l[0]
        cv2.line(mask, (x1,y1), (x2,y2), 255, 2)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    epsilon = 0.02*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) < 4:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float32)
    else:
        hull = cv2.convexHull(approx)
        corners = hull.reshape(-1,2).astype(np.float32)
        if len(corners) > 4:
            rect = cv2.minAreaRect(hull)
            corners = cv2.boxPoints(rect).astype(np.float32)

    return _order_corners(corners)

def warp_board(bgr, cfg):
    w = cfg["board"]["warp_size"]
    src = _find_board_corners(bgr, cfg)
    dst = np.float32([[0,0],[w,0],[w,w],[0,w]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (w,w))
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
    return cells  # [((r,c), BGR)]
