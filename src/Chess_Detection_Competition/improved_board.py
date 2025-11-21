# ===== FIX 1: Improve Homography + Add Rotation Correction =====
import cv2
import numpy as np
import os

# Global variables for mouse callback
_selected_points = []
_temp_img = None
_display_img = None
_scale_factor = 1.0

def _mouse_callback(event, x, y, flags, param):
    """Mouse callback for manual corner selection"""
    global _selected_points, _temp_img, _display_img, _scale_factor
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(_selected_points) < 4:
            real_x = int(x / _scale_factor)
            real_y = int(y / _scale_factor)
            
            _selected_points.append((real_x, real_y))
            print(f"Point {len(_selected_points)}: ({real_x}, {real_y})")
            
            cv2.circle(_temp_img, (real_x, real_y), 10, (0, 255, 0), -1)
            cv2.putText(_temp_img, str(len(_selected_points)), (real_x+15, real_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            _display_img = cv2.resize(_temp_img, None, fx=_scale_factor, fy=_scale_factor)
            cv2.imshow("Select 4 Corners (TL, TR, BR, BL)", _display_img)


def select_corners_manually(bgr, max_display_height=800):
    """Manual corner selection with display scaling"""
    global _selected_points, _temp_img, _display_img, _scale_factor
    
    _selected_points = []
    _temp_img = bgr.copy()
    
    h, w = bgr.shape[:2]
    if h > max_display_height:
        _scale_factor = max_display_height / h
    else:
        _scale_factor = 1.0
    
    _display_img = cv2.resize(_temp_img, None, fx=_scale_factor, fy=_scale_factor)
    
    cv2.namedWindow("Select 4 Corners (TL, TR, BR, BL)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select 4 Corners (TL, TR, BR, BL)", _mouse_callback)
    
    print("\n=== Manual Corner Selection ===")
    print("Click 4 corners: TL -> TR -> BR -> BL")
    print("Press 'r' to reset, 'q' when done")
    
    while True:
        cv2.imshow("Select 4 Corners (TL, TR, BR, BL)", _display_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            _selected_points = []
            _temp_img = bgr.copy()
            _display_img = cv2.resize(_temp_img, None, fx=_scale_factor, fy=_scale_factor)
            print("Reset!")
        
        elif key == ord('q') and len(_selected_points) == 4:
            break
        
        elif key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Cancelled")
    
    cv2.destroyAllWindows()
    
    corners = np.array(_selected_points, dtype=np.float32)
    return corners


def _order_corners(pts4):
    """Sort corners: TL, TR, BR, BL"""
    pts = np.asarray(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _auto_detect_corners(bgr):
    """Advanced auto corner detection with multiple strategies"""
    h, w = bgr.shape[:2]
    
    # Strategy 1: HSV Color Masking (works well for wooden/colored boards)
    try:
        corners = _detect_board_hsv_mask(bgr)
        if corners is not None and _validate_corners(corners, w, h):
            corners = _order_corners(corners)
            print("[auto] Strategy 1: HSV Masking SUCCESS")
            return corners
    except Exception as e:
        print(f"[auto] HSV masking failed: {e}")
    
    # Strategy 2: Enhanced Edge + Hough Lines
    try:
        corners = _detect_board_edges_hough(bgr)
        if corners is not None and _validate_corners(corners, w, h):
            corners = _order_corners(corners)
            print("[auto] Strategy 2: Edge+Hough SUCCESS")
            return corners
    except Exception as e:
        print(f"[auto] Edge+Hough failed: {e}")
    
    # Strategy 3: Improved Contour Detection
    try:
        corners = _detect_board_contours_improved(bgr)
        if corners is not None and _validate_corners(corners, w, h):
            corners = _order_corners(corners)
            print("[auto] Strategy 3: Improved Contours SUCCESS")
            return corners
    except Exception as e:
        print(f"[auto] Contour detection failed: {e}")
    
    # Strategy 4: Conservative crop (fallback)
    print("[auto] Strategy 4: Conservative crop (fallback)")
    margin = int(0.08 * min(h, w))
    corners = np.array([
        [margin, margin],
        [w - margin, margin],
        [w - margin, h - margin],
        [margin, h - margin]
    ], dtype=np.float32)
    
    return corners

def _detect_board_hsv_mask(bgr):
    """Detect chessboard using HSV color masking"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]
    
    # HSV ranges for different board colors
    ranges = [
        ([5, 20, 80], [25, 100, 220]),    # Light wooden boards
        ([0, 15, 40], [20, 80, 120]),     # Dark wooden boards  
        ([0, 0, 150], [180, 30, 255]),    # White/cream boards
        ([35, 40, 40], [85, 255, 200])   # Green tournament boards
    ]
    
    best_mask = None
    best_area = 0
    
    for (lower, upper) in ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area > (h * w * 0.2) and area > best_area:
                best_area = area
                best_mask = mask
    
    if best_mask is None:
        return None
        
    # Find board contour from best mask
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    largest = max(contours, key=cv2.contourArea)
    
    # Approximate to quadrilateral
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    
    # If not 4 points, use bounding rect
    x, y, w_rect, h_rect = cv2.boundingRect(largest)
    return np.array([
        [x, y],
        [x + w_rect, y], 
        [x + w_rect, y + h_rect],
        [x, y + h_rect]
    ], dtype=np.float32)

def _detect_board_edges_hough(bgr):
    """Detect chessboard using enhanced edge detection and Hough lines"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Enhanced edge detection with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Adaptive Canny thresholds
    median = np.median(filtered)
    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))
    edges = cv2.Canny(filtered, lower, upper)
    
    # Hough line detection
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=80,
        minLineLength=min(h, w) * 0.3, maxLineGap=20
    )
    
    if lines is None:
        return None
    
    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x2 - x1 != 0:
            angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
        else:
            angle = 90
        
        if angle < 10 or angle > 170:  # Horizontal
            h_lines.append((x1, y1, x2, y2))
        elif 80 < angle < 100:  # Vertical
            v_lines.append((x1, y1, x2, y2))
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None
    
    # Find extreme lines
    h_lines = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
    v_lines = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)
    
    top_line = h_lines[0]
    bottom_line = h_lines[-1] 
    left_line = v_lines[0]
    right_line = v_lines[-1]
    
    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
            
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        return (x, y)
    
    # Find corner intersections
    tl = line_intersection(top_line, left_line)
    tr = line_intersection(top_line, right_line)
    bl = line_intersection(bottom_line, left_line)
    br = line_intersection(bottom_line, right_line)
    
    if None in [tl, tr, bl, br]:
        return None
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _detect_board_contours_improved(bgr):
    """Improved contour-based detection"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Try multiple preprocessing approaches
    approaches = [
        cv2.GaussianBlur(gray, (5, 5), 0),
        cv2.medianBlur(gray, 5),
        cv2.bilateralFilter(gray, 9, 75, 75)
    ]
    
    for processed in approaches:
        edges = cv2.Canny(processed, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:10]:
            area = cv2.contourArea(contour)
            
            if area < (h * w) * 0.15:  # Lowered threshold
                continue
                
            # Try different epsilon values for approximation
            for eps_factor in [0.01, 0.02, 0.03, 0.05]:
                epsilon = eps_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    return approx.reshape(4, 2).astype(np.float32)
    
    return None

def _validate_corners(corners, img_w, img_h):
    """Validate that corners form a reasonable quadrilateral"""
    if corners is None or len(corners) != 4:
        return False
    
    # All corners should be within image bounds (with small margin)
    margin = 5
    for x, y in corners:
        if x < -margin or x >= img_w + margin or y < -margin or y >= img_h + margin:
            return False
    
    # Calculate area - should be reasonable size
    area = cv2.contourArea(corners)
    min_area = (img_w * img_h) * 0.1  # At least 10% of image
    max_area = (img_w * img_h) * 0.95  # At most 95% of image
    
    if area < min_area or area > max_area:
        return False
    
    return True


def warp_board_v2(bgr, cfg, manual_mode=False, video_name=None):
    """Perspective warp with fallback"""
    w = cfg["board"]["warp_size"]
    
    if manual_mode:
        src = select_corners_manually(bgr, max_display_height=800)
    else:
        src = _auto_detect_corners(bgr)
    
    dst = np.float32([[0, 0], [w, 0], [w, w], [0, w]])
    M = cv2.getPerspectiveTransform(src, dst)
    
    warped = cv2.warpPerspective(
        bgr, M, (w, w),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128)
    )
    
    return warped, M


def split_grid_v2_debug(warped, cell_px):
    """Grid splitting - Trivial split method (matches LiveChess2FEN)"""
    H, W = warped.shape[:2]

    # No outer margin - direct division like LiveChess2FEN
    # Reference: LiveChess2FEN/lc2fen/split_board.py:split_board_image_trivial()
    margin = 0  # Changed from 0.05 to match LiveChess2FEN
    xs = np.linspace(margin, W - margin, 9)
    ys = np.linspace(margin, H - margin, 9)
    
    cells = _extract_cells(warped, xs, ys, cell_px)
    visualize_grid_detection(warped, xs, ys, "debug/grid_overlay.jpg")
    
    return cells, xs, ys


def _extract_cells(warped, xs, ys, cell_px):
    """Extract 64 cells"""
    H, W = warped.shape[:2]
    cells = []
    
    for r in range(8):
        for c in range(8):
            x0, x1 = int(xs[c]), int(xs[c+1])
            y0, y1 = int(ys[r]), int(ys[r+1])
            
            margin = 3
            x0 += margin
            x1 -= margin
            y0 += margin
            y1 -= margin
            
            x0 = max(0, min(x0, W-1))
            x1 = max(x0+1, min(x1, W))
            y0 = max(0, min(y0, H-1))
            y1 = max(y0+1, min(y1, H))
            
            patch = warped[y0:y1, x0:x1]
            
            if patch.size == 0:
                patch = np.zeros((cell_px, cell_px, 3), dtype=np.uint8)
            else:
                patch = cv2.resize(patch, (cell_px, cell_px), interpolation=cv2.INTER_AREA)
            
            cells.append(((r, c), patch))
    
    return cells


def visualize_grid_detection(warped, xs, ys, save_path="debug/grid.jpg"):
    """Visualize grid"""
    vis = warped.copy()
    H, W = vis.shape[:2]
    
    for x in xs:
        cv2.line(vis, (int(x), 0), (int(x), H), (0, 255, 0), 2)
    
    for y in ys:
        cv2.line(vis, (0, int(y)), (W, int(y)), (255, 0, 0), 2)
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, vis)
    return vis


def split_grid_v2(warped, cell_px):
    cells, _, _ = split_grid_v2_debug(warped, cell_px)
    return cells

