import cv2
import os

def extract_frames(video_path, output_folder, frame_skip=10):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            filename = os.path.join(output_folder, f"frame_{saved}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"[INFO] Saved {saved} frames from {video_path} to {output_folder}")
