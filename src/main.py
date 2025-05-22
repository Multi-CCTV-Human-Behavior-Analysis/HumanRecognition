import cv2
import os
import numpy as np
import random
from feature_extraction.feature_extractor import TorchReID
from detection.yolo_model import load_yolo_model
from tracking.deepsort import DeepSORT

def get_random_color():
    """Generate a random color."""
    return tuple(random.randint(0, 255) for _ in range(3))

def save_crop(frame, bbox, output_dir, track_id, frame_id, video_tag):
    os.makedirs(output_dir, exist_ok=True)
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        path = f"{output_dir}/{video_tag}_id{track_id}_f{frame_id}.jpg"
        cv2.imwrite(path, crop)
        return path
    return None

def extract_feature_batch(encoder_obj, crop_paths):
    features = []
    for path in crop_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (64, 128))  # Reshape for MARS model
            img = np.expand_dims(img, axis=0).astype(np.uint8)  # Shape: (1, 128, 64, 3)
            feat = encoder_obj(img)[0]  # Directly call ImageEncoder.__call__()
            features.append((path, feat))
    return features

def main():
    yolo_model = load_yolo_model('yolov8n.pt')
    encoder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/mars-small128.pb'))
    deepsort1 = DeepSORT(encoder_path)
    deepsort2 = DeepSORT(encoder_path)
    reid_model = TorchReID(model_path='models/resnet50-19c8e357.pth', gpu_id=0) # Adjust GPU ID as needed

    video_capture1 = cv2.VideoCapture('test_video/view-HC2.mp4')
    video_capture2 = cv2.VideoCapture('test_video/view-HC3.mp4')

    ret1, frame1 = video_capture1.read()
    ret2, frame2 = video_capture2.read()
    if not ret1 or not ret2:
        print("Error: Unable to read video files.")
        return

    out1 = cv2.VideoWriter('output1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame1.shape[1], frame1.shape[0]))
    out2 = cv2.VideoWriter('output2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame2.shape[1], frame2.shape[0]))

    id_to_crop1, id_to_crop2 = {}, {}
    frame_data1, frame_data2 = {}, {}
    frame_id = 0

    # Dictionaries to store colors for each track_id
    colors1 = {}
    colors2 = {}

    while True:
        ret1, frame1 = video_capture1.read()
        ret2, frame2 = video_capture2.read()
        if not ret1 or not ret2:
            break

        # Video 1
        boxes1 = yolo_model.detect_people(frame1)
        tracks1 = deepsort1.update(frame1, boxes1)
        frame_data1[frame_id] = {'boxes': [], 'ids': []}
        for (bbox, track_id) in tracks1:
            if track_id not in colors1:
                colors1[track_id] = get_random_color()  # Assign a random color
            color = colors1[track_id]
            cv2.rectangle(frame1, tuple(map(int, bbox[:2])), tuple(map(int, bbox[2:])), color, 2)
            # Adjust text position to be inside the bounding box
            text_x = int(bbox[0]) + 5  # Slightly offset from the left edge
            text_y = int(bbox[1]) + 20  # Slightly offset from the top edge
            cv2.putText(frame1, f"ID {track_id}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            path = save_crop(frame1, bbox, "crops/video1", track_id, frame_id, "v1")
            if path:
                id_to_crop1.setdefault(track_id, []).append(path)
            frame_data1[frame_id]['boxes'].append(bbox)
            frame_data1[frame_id]['ids'].append(track_id)

        # Video 2
        boxes2 = yolo_model.detect_people(frame2)
        tracks2 = deepsort2.update(frame2, boxes2)
        frame_data2[frame_id] = {'boxes': [], 'ids': []}
        for (bbox, track_id) in tracks2:
            if track_id not in colors2:
                colors2[track_id] = get_random_color()  # Assign a random color
            color = colors2[track_id]
            cv2.rectangle(frame2, tuple(map(int, bbox[:2])), tuple(map(int, bbox[2:])), color, 2)
            # Adjust text position to be inside the bounding box
            text_x = int(bbox[0]) + 5  # Slightly offset from the left edge
            text_y = int(bbox[1]) + 20  # Slightly offset from the top edge
            cv2.putText(frame2, f"ID {track_id}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            path = save_crop(frame2, bbox, "crops/video2", track_id, frame_id, "v2")
            if path:
                id_to_crop2.setdefault(track_id, []).append(path)
            frame_data2[frame_id]['boxes'].append(bbox)
            frame_data2[frame_id]['ids'].append(track_id)

        out1.write(frame1)
        out2.write(frame2)
        frame_id += 1

        cv2.imshow("Video 1", frame1)
        cv2.imshow("Video 2", frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture1.release()
    video_capture2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

    # ReID Matching Using TorchReID
    # Aggregate all crops for each ID and extract features
    features_vid1 = {track_id: reid_model.extract_features(paths) for track_id, paths in id_to_crop1.items()}
    features_vid2 = {track_id: reid_model.extract_features(paths) for track_id, paths in id_to_crop2.items()}

    # Match using Euclidean distance and aggregate feature comparison
    matched_ids = reid_model.match_tracks(features_vid1, features_vid2, threshold=320)

    # Save matched ID pairs
    print("Matched IDs:", matched_ids)
    with open("matched_ids.txt", "w") as f:
        for k, v in matched_ids.items():
            f.write(f"{k} -> {v}\n")

if __name__ == "__main__":
    main()