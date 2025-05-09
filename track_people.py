import os
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Load YOLO model (you can replace with yolov8s.pt for higher accuracy)
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

def track_people(video_path, save_output=False, output_path="tracked_output.mp4"):
    """
    Tracks multiple people in a video using YOLOv8 and DeepSORT, and displays live tracking.

    Args:
        video_path (str): Path to the video file.
        save_output (bool): If True, saves the video with tracking overlay.
        output_path (str): Path to save the tracked video (only if save_output=True).
    """
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = 0

    # Optional video writer
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Tracking started on: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        # Extract person detections
        person_detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        # Update DeepSORT tracker
        tracks = tracker.update_tracks(person_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'P{track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if save_output:
            out.write(frame)

        cv2.imshow("Multi-person Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()
    print("[✅] Tracking completed.")
