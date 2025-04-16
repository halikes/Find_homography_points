import cv2
import os

cap = cv2.VideoCapture("data/sample_video_002_7s.mp4")
os.makedirs("colmap_frames", exist_ok=True)
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"colmap_frames/frame_{frame_idx:05d}.jpg", frame)
    frame_idx += 1
cap.release()

