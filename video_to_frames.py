import cv2
import os

def extract_frames(video_path, output_folder, frame_skip=10):
    """
    Extracts frames from a video file and saves them to the output folder.
    Skips every `frame_skip` frames for efficiency.
    
    Args:
        video_path (str): Path to the input video.
        output_folder (str): Directory where frames will be saved.
        frame_skip (int): Number of frames to skip between saves (default is 10).
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_count = 0

    for count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

    cap.release()
    print(f"[✅] {saved_count} frames saved from '{os.path.basename(video_path)}' → '{output_folder}'")
