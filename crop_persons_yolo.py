import os
import cv2
from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt' for better accuracy

def crop_persons_from_frames(frames_folder, output_folder, conf_threshold=0.4):
    os.makedirs(output_folder, exist_ok=True)
    frame_files = [f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))]

    for frame_file in sorted(frame_files):
        img_path = os.path.join(frames_folder, frame_file)
        img = cv2.imread(img_path)
        results = model(img)[0]

        count = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > conf_threshold:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = img[y1:y2, x1:x2]
                resized = cv2.resize(cropped, (128, 256))
                save_name = f"{os.path.splitext(frame_file)[0]}_p{count}.jpg"
                save_path = os.path.join(output_folder, save_name)
                cv2.imwrite(save_path, resized)
                count += 1
