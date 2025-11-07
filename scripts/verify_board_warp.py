import os, glob, cv2
from Chess_Detection_Competition.utils import load_config, ensure_dir
from Chess_Detection_Competition.board import warp_board


cfg = load_config()
ensure_dir("debug")

videos = glob.glob(os.path.join(cfg["paths"]["videos_dir"], "*.mp4"))
assert videos, "ใส่วิดีโอไว้ที่ data/public/videos/*.mp4 ก่อน"

v = videos[0]
cap = cv2.VideoCapture(v)
ok, frame = cap.read()
cap.release()
assert ok, f"อ่านเฟรมแรกไม่ได้: {v}"

warped, _ = warp_board(frame, cfg)
cv2.imwrite("debug/first_frame.jpg", frame)
cv2.imwrite("debug/warped.jpg", warped)
print("Saved debug/first_frame.jpg และ debug/warped.jpg")
