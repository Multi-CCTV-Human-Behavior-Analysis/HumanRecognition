import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model (choose yolov8s.pt for better accuracy)
model = YOLO('yolov8n.pt')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

def track_and_crop(video_path, output_root="tracked_crops", conf_threshold=0.4):
    """
    Tracks people in a video using YOLOv8 + DeepSORT, crops them frame by frame,
    and saves crops into separate folders (one per person ID).

    Args:
        video_path (str): Input video path.
        output_root (str): Root folder to save person crops.
        conf_threshold (float): YOLO confidence threshold for person detection.
    """
    os.makedirs(output_root, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    total_crops = 0
    unique_ids = set()

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose = False)[0]
        person_detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width, height = x2 - x1, y2 - y1
                person_detections.append(([x1, y1, width, height], conf, 'person'))

        # Track across frames
        tracks = tracker.update_tracks(person_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            unique_ids.add(track_id)
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            person_dir = os.path.join(output_root, f"P{track_id}")
            os.makedirs(person_dir, exist_ok=True)

            save_path = os.path.join(person_dir, f"frame_{frame_idx}.jpg")
            try:
                crop_resized = cv2.resize(crop, (128, 256))
                cv2.imwrite(save_path, crop_resized)
                total_crops += 1
            except Exception:
                continue

        frame_idx += 1

    cap.release()
    print(f"[✅] Tracking complete: {total_crops} crops saved across {len(unique_ids)} people in '{output_root}'")
