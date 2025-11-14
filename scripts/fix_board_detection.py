#!/usr/bin/env python3
"""
Improve board detection using multiple strategies including HSV masking and contour detection.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def detect_board_hsv_mask(bgr, debug=False):
    """
    Detect chessboard using HSV color masking to find the board area
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]
    
    # Multiple HSV ranges for different board colors
    ranges = [
        # Light wooden boards
        ([5, 20, 80], [25, 100, 220]),    
        # Dark wooden boards  
        ([0, 15, 40], [20, 80, 120]),
        # White/cream boards
        ([0, 0, 150], [180, 30, 255]),
        # Green tournament boards
        ([35, 40, 40], [85, 255, 200])
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
            
            # Must be reasonable size (at least 20% of image)
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
        if debug:
            print("[HSV] Found 4-point contour ✓")
        return approx.reshape(4, 2).astype(np.float32)
    
    # If not 4 points, use bounding rect
    x, y, w_rect, h_rect = cv2.boundingRect(largest)
    
    if debug:
        print(f"[HSV] Using bounding rect: {w_rect}x{h_rect}")
    
    return np.array([
        [x, y],
        [x + w_rect, y], 
        [x + w_rect, y + h_rect],
        [x, y + h_rect]
    ], dtype=np.float32)

def detect_board_edges_hough(bgr, debug=False):
    """
    Detect chessboard using edge detection and Hough line transform
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Enhanced edge detection
    # 1. CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 2. Bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 3. Canny edge detection with automatic thresholding
    median = np.median(filtered)
    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))
    edges = cv2.Canny(filtered, lower, upper)
    
    if debug:
        cv2.imwrite("debug/edges.jpg", edges)
    
    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=min(h, w) * 0.3,
        maxLineGap=20
    )
    
    if lines is None:
        if debug:
            print("[Hough] No lines detected")
        return None
    
    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        if x2 - x1 != 0:
            angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
        else:
            angle = 90
        
        if angle < 10 or angle > 170:  # Horizontal
            h_lines.append((x1, y1, x2, y2))
        elif 80 < angle < 100:  # Vertical
            v_lines.append((x1, y1, x2, y2))
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        if debug:
            print(f"[Hough] Not enough lines: h={len(h_lines)}, v={len(v_lines)}")
        return None
    
    # Find extreme lines
    h_lines = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)  # Sort by y
    v_lines = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)  # Sort by x
    
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
        if debug:
            print("[Hough] Failed to find all intersections")
        return None
    
    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    
    if debug:
        print(f"[Hough] Found corners: {corners}")
    
    return corners

def detect_board_contours(bgr, debug=False):
    """
    Detect chessboard using contour detection (original method)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection  
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        
        # Must be reasonable size
        if area < (h * w) * 0.2:
            continue
            
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            if debug:
                print(f"[Contour] Found 4-sided contour, area={area}")
            return approx.reshape(4, 2).astype(np.float32)
    
    return None

def improved_board_detection(bgr, debug=False):
    """
    Multi-strategy board detection with fallbacks
    """
    h, w = bgr.shape[:2]
    
    if debug:
        print(f"🔍 Detecting board in {w}x{h} image")
        os.makedirs("debug", exist_ok=True)
    
    strategies = [
        ("HSV Masking", detect_board_hsv_mask),
        ("Edge+Hough", detect_board_edges_hough), 
        ("Contours", detect_board_contours)
    ]
    
    for name, detect_func in strategies:
        if debug:
            print(f"  Trying {name}...")
            
        try:
            corners = detect_func(bgr, debug=debug)
            
            if corners is not None:
                # Validate corners
                if validate_corners(corners, w, h):
                    corners = order_corners(corners)
                    if debug:
                        print(f"  ✅ {name} succeeded")
                        # Save detection visualization
                        vis = bgr.copy()
                        for i, (x, y) in enumerate(corners):
                            cv2.circle(vis, (int(x), int(y)), 10, (0, 255, 0), -1)
                            cv2.putText(vis, str(i+1), (int(x)+15, int(y)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imwrite(f"debug/detected_corners_{name.lower().replace('+', '_')}.jpg", vis)
                    return corners
                else:
                    if debug:
                        print(f"  ❌ {name} - invalid corners")
            else:
                if debug:
                    print(f"  ❌ {name} - no corners found")
                    
        except Exception as e:
            if debug:
                print(f"  ❌ {name} - error: {e}")
    
    # Final fallback - conservative crop
    if debug:
        print("  🔧 Using conservative crop fallback")
        
    margin = int(0.1 * min(h, w))
    corners = np.array([
        [margin, margin],
        [w - margin, margin],
        [w - margin, h - margin], 
        [margin, h - margin]
    ], dtype=np.float32)
    
    return corners

def validate_corners(corners, img_w, img_h):
    """Validate that corners form a reasonable quadrilateral"""
    if corners is None or len(corners) != 4:
        return False
    
    # All corners should be within image bounds
    for x, y in corners:
        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            return False
    
    # Calculate area - should be reasonable size
    area = cv2.contourArea(corners)
    min_area = (img_w * img_h) * 0.1  # At least 10% of image
    max_area = (img_w * img_h) * 0.9  # At most 90% of image
    
    if area < min_area or area > max_area:
        return False
    
    # Check if quadrilateral is roughly convex
    # (no angles should be too sharp)
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]
        
        # Calculate angle at p2
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        # Angle should be reasonable (30-150 degrees)
        if angle < 30 or angle > 150:
            return False
    
    return True

def order_corners(corners):
    """Order corners as [top-left, top-right, bottom-right, bottom-left]"""
    corners = np.array(corners, dtype=np.float32)
    
    # Calculate center
    center = corners.mean(axis=0)
    
    # Sort by angle from center
    def angle_from_center(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])
    
    # Find top-left (smallest angle)
    angles = [angle_from_center(p) for p in corners]
    sorted_indices = np.argsort(angles)
    
    # Rearrange to start from top-left going clockwise
    ordered = []
    for i in sorted_indices:
        ordered.append(corners[i])
    
    # Ensure we have TL, TR, BR, BL order
    ordered = np.array(ordered)
    
    # Simple ordering: sum of coordinates for TL/BR, difference for TR/BL
    sums = ordered.sum(axis=1)
    diffs = np.diff(ordered, axis=1).flatten()
    
    tl = ordered[np.argmin(sums)]      # Smallest x+y
    br = ordered[np.argmax(sums)]      # Largest x+y  
    tr = ordered[np.argmin(diffs)]     # Smallest x-y
    bl = ordered[np.argmax(diffs)]     # Largest x-y
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

if __name__ == "__main__":
    # Test the improved detection
    ROOT = Path(".").resolve()
    videos_dir = ROOT / "data/public/videos"
    
    if not videos_dir.exists():
        print(f"❌ Videos directory not found: {videos_dir}")
        exit(1)
    
    videos = list(videos_dir.glob("*.mp4"))
    if not videos:
        print(f"❌ No videos found in: {videos_dir}")
        exit(1)
    
    print(f"🎬 Testing improved board detection on {len(videos)} videos")
    
    for video_path in videos:
        print(f"\n📹 Testing: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        
        if not ok:
            print(f"❌ Cannot read frame from {video_path.name}")
            continue
        
        # Test detection
        corners = improved_board_detection(frame, debug=True)
        
        if corners is not None:
            print(f"✅ Board detected successfully")
            print(f"   Corners: {corners}")
            
            # Test warping
            w = 800
            dst = np.float32([[0, 0], [w, 0], [w, w], [0, w]])
            M = cv2.getPerspectiveTransform(corners, dst)
            warped = cv2.warpPerspective(frame, M, (w, w))
            
            cv2.imwrite(f"debug/warped_{video_path.stem}.jpg", warped)
            print(f"   Warped board saved to debug/warped_{video_path.stem}.jpg")
        else:
            print(f"❌ Failed to detect board")
    
    print("\n✅ Board detection testing complete!")