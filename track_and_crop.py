import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Init YOLO model
model = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy

# Init DeepSort
tracker = DeepSort(max_age=30)

def track_and_crop(video_path, output_root="tracked_crops", conf_threshold=0.4):
    os.makedirs(output_root, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = model(frame)[0]
        person_detections = []

        for box in detections.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width, height = x2 - x1, y2 - y1
                person_detections.append(([x1, y1, width, height], conf, 'person'))

        # Update tracker with detections
        tracks = tracker.update_tracks(person_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # Crop and save person image
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            person_dir = os.path.join(output_root, f"P{track_id}")
            os.makedirs(person_dir, exist_ok=True)

            save_path = os.path.join(person_dir, f"frame_{frame_idx}.jpg")
            crop_resized = cv2.resize(crop, (128, 256))  # For OSNet compatibility
            cv2.imwrite(save_path, crop_resized)

        frame_idx += 1

    cap.release()
    print(f"[INFO] Tracking complete. Cropped images saved to: {output_root}")
