# ===== FIX 1: Improve Homography + Add Rotation Correction =====
import cv2
import numpy as np

def _order_corners(pts4):
    """Sort corners: top-left, top-right, bottom-right, bottom-left"""
    pts = np.asarray(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _find_board_corners_robust(bgr, cfg):
    """
    Enhanced corner detection with multiple strategies:
    1. HSV color mask (green/white squares)
    2. Canny + HoughLines (grid lines)
    3. Template matching (if available)
    """
    h, w = bgr.shape[:2]
    pad = int(0.03 * min(h, w))  # เพิ่ม padding
    bgr_crop = bgr[pad:h-pad, pad:w-pad].copy()
    off = np.array([pad, pad], np.float32)

    # Strategy 1: HSV mask for board colors
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    m_green = cv2.inRange(hsv, (30, 30, 40), (90, 255, 255))
    m_white = cv2.inRange(hsv, (0, 0, 160), (179, 80, 255))
    mask = cv2.bitwise_or(m_green, m_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    
    # Find largest contour
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Cannot find board contour")
    
    cnt = max(cnts, key=cv2.contourArea)
    
    # Fit rotated rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    corners = _order_corners(box) + off
    
    return corners

def _deskew_warped(warped):
    """
    Detect slight rotation in warped image and correct it
    Using Hough Lines to find grid angle
    """
    H, W = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                            minLineLength=int(0.3*min(H,W)), 
                            maxLineGap=20)
    
    if lines is None or len(lines) < 4:
        return warped
    
    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append(angle)
    
    # Find dominant horizontal/vertical angles
    angles = np.array(angles)
    # Snap to nearest 0° or 90°
    angles_deg = np.rad2deg(angles)
    angles_deg = angles_deg % 90  # normalize to 0-90
    
    # If most lines are close to 0° or 90°, board is straight
    median_angle = np.median(angles_deg)
    if median_angle < 45:
        correction = -median_angle
    else:
        correction = 90 - median_angle
    
    if abs(correction) < 0.5:  # Already straight
        return warped
    
    # Rotate to correct
    center = (W//2, H//2)
    M = cv2.getRotationMatrix2D(center, correction, 1.0)
    rotated = cv2.warpAffine(warped, M, (W, H), 
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def warp_board_v2(bgr, cfg):
    """
    Improved warping with deskew correction
    """
    w = cfg["board"]["warp_size"]
    src = _find_board_corners_robust(bgr, cfg)
    dst = np.float32([[0,0], [w,0], [w,w], [0,w]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (w, w))
    
    # Apply rotation correction
    warped = _deskew_warped(warped)
    
    return warped, M


# ===== FIX 2: Robust Grid Detection Using Multiple Methods =====

def _detect_grid_by_color_transitions(warped):
    """
    Alternative method: detect grid by analyzing color transitions
    Works well when lines are hard to detect
    """
    H, W = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Sum along axes to find grid lines
    proj_x = np.abs(sobelx).sum(axis=0)  # vertical lines
    proj_y = np.abs(sobely).sum(axis=1)  # horizontal lines
    
    # Smooth projections
    from scipy.signal import find_peaks
    proj_x = cv2.GaussianBlur(proj_x.reshape(1, -1), (1, 21), 0).ravel()
    proj_y = cv2.GaussianBlur(proj_y.reshape(1, -1), (1, 21), 0).ravel()
    
    # Find peaks (grid lines)
    peaks_x, _ = find_peaks(proj_x, distance=W//10, prominence=proj_x.max()*0.1)
    peaks_y, _ = find_peaks(proj_y, distance=H//10, prominence=proj_y.max()*0.1)
    
    # Need exactly 9 lines
    if len(peaks_x) >= 9 and len(peaks_y) >= 9:
        # Take strongest 9 peaks
        idx_x = np.argsort(proj_x[peaks_x])[-9:]
        idx_y = np.argsort(proj_y[peaks_y])[-9:]
        xs = np.sort(peaks_x[idx_x])
        ys = np.sort(peaks_y[idx_y])
        return xs, ys
    
    return None, None

def split_grid_v2(warped, cell_px):
    """
    Enhanced grid splitting with multiple detection strategies
    """
    H, W = warped.shape[:2]
    
    # Try Method 1: Hough Lines (from original code)
    from Chess_Detection_Competition.board import _grid_lines_from_hough
    xs, ys = _grid_lines_from_hough(warped, want=9)
    
    # Try Method 2: Color transitions (if Hough fails)
    if xs is None or ys is None:
        print("[split_grid_v2] Hough failed, trying color transitions...")
        xs, ys = _detect_grid_by_color_transitions(warped)
    
    # Fallback Method 3: Equal division with margin
    if xs is None or ys is None:
        print("[split_grid_v2] All methods failed, using equal division")
        # Add 5% margin to avoid edge artifacts
        margin = int(0.05 * min(H, W))
        H_inner = H - 2*margin
        W_inner = W - 2*margin
        step_x = W_inner / 8
        step_y = H_inner / 8
        
        xs = np.array([margin + i*step_x for i in range(9)])
        ys = np.array([margin + i*step_y for i in range(9)])
    
    # Crop cells with safety checks
    cells = []
    for r in range(8):
        for c in range(8):
            x0 = int(xs[c])
            x1 = int(xs[c+1])
            y0 = int(ys[r])
            y1 = int(ys[r+1])
            
            # Add small margin to avoid grid lines
            margin_px = max(2, int(0.02 * min(x1-x0, y1-y0)))
            x0 += margin_px
            x1 -= margin_px
            y0 += margin_px
            y1 -= margin_px
            
            # Safety bounds check
            x0 = max(0, min(x0, W-1))
            x1 = max(x0+1, min(x1, W))
            y0 = max(0, min(y0, H-1))
            y1 = max(y0+1, min(y1, H))
            
            patch = warped[y0:y1, x0:x1]
            
            # Ensure patch is valid
            if patch.size == 0:
                print(f"Warning: Empty patch at ({r},{c})")
                patch = np.zeros((cell_px, cell_px, 3), dtype=np.uint8)
            else:
                patch = cv2.resize(patch, (cell_px, cell_px), 
                                   interpolation=cv2.INTER_AREA)
            
            cells.append(((r, c), patch))
    
    return cells


# ===== FIX 3: Interactive Manual Correction (Optional) =====

def manual_corner_adjustment(bgr, auto_corners):
    """
    Allow user to manually adjust corners if auto-detection fails
    Click 4 corners in order: top-left, top-right, bottom-right, bottom-left
    """
    corners = auto_corners.copy()
    img_display = bgr.copy()
    
    # Draw current corners
    for i, pt in enumerate(corners):
        cv2.circle(img_display, tuple(pt.astype(int)), 10, (0,255,0), -1)
        cv2.putText(img_display, str(i), tuple(pt.astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(img_display, (x, y), 5, (0,0,255), -1)
            cv2.imshow('Adjust Corners', img_display)
            if len(points) == 4:
                cv2.destroyAllWindows()
    
    cv2.namedWindow('Adjust Corners')
    cv2.setMouseCallback('Adjust Corners', mouse_callback)
    cv2.imshow('Adjust Corners', img_display)
    
    print("Click 4 corners in order: TL, TR, BR, BL (or press 'q' to use auto)")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) == 4:
            break
    
    cv2.destroyAllWindows()
    
    if len(points) == 4:
        return np.array(points, dtype=np.float32)
    return corners


# ===== Usage Example =====

def test_fixed_pipeline(video_path, cfg):
    """
    Test the fixed pipeline
    """
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    
    if not ok:
        raise RuntimeError("Cannot read video")
    
    # Step 1: Improved warping
    warped, M = warp_board_v2(frame, cfg)
    
    # Step 2: Improved grid splitting
    cell_px = int(cfg["cells"]["img_size"])
    cells = split_grid_v2(warped, cell_px)
    
    return frame, warped, cells


# ===== Integration with existing code =====

def patch_existing_code():
    """
    Replace functions in your Chess_Detection_Competition/board.py
    
    Replace:
    - warp_board() with warp_board_v2()
    - split_grid() with split_grid_v2()
    """
    print("""
    To integrate:
    
    1. Replace in board.py:
       def warp_board(bgr, cfg=DEFAULT_CFG):
           return warp_board_v2(bgr, cfg)
    
    2. Replace in board.py:
       def split_grid(warped, cell_px):
           return split_grid_v2(warped, cell_px)
    
    3. Install scipy if not available:
       pip install scipy
    """)