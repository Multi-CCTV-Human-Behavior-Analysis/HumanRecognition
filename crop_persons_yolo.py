import os
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can upgrade to 'yolov8s.pt' if needed

def crop_persons_from_frames(frames_folder, output_folder, conf_threshold=0.4):
    """
    Detects and crops person regions from each frame using YOLOv8,
    resizes them to 128x256, and saves into the output folder.

    Args:
        frames_folder (str): Folder containing input video frames.
        output_folder (str): Folder to save cropped person images.
        conf_threshold (float): Confidence threshold for person detection.
    """
    os.makedirs(output_folder, exist_ok=True)
    frame_files = [f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.png'))]

    total_saved = 0

    for frame_file in sorted(frame_files):
        img_path = os.path.join(frames_folder, frame_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            results = model(img, verbose = False)[0]
            count = 0

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 0 and conf > conf_threshold:  # Class 0 = person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped = img[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue
                    resized = cv2.resize(cropped, (128, 256))
                    save_name = f"{os.path.splitext(frame_file)[0]}_p{count}.jpg"
                    save_path = os.path.join(output_folder, save_name)
                    cv2.imwrite(save_path, resized)
                    count += 1
                    total_saved += 1
        except Exception:
            continue

    print(f"[✅] Saved {total_saved} person crops to '{output_folder}'")
