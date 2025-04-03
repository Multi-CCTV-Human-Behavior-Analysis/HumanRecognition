import cv2
import os

# Replace with RTSP URL (or use a .mp4 file for now)
# Example: 'rtsp://admin:password@192.168.1.101:554/Streaming/Channels/101'
video_path = 'your_video.mp4'

cap = cv2.VideoCapture(video_path)

# Frame save directory
save_dir = 'frames'
os.makedirs(save_dir, exist_ok=True)

frame_count = 0
frame_skip = 10  # Capture every 10th frame to avoid duplicates

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        frame_filename = os.path.join(save_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()
print(f"Saved {frame_count // frame_skip} frames to {save_dir}")
