import cv2
import os
import numpy as np
import random
from detection.yolo_model import load_yolo_model
from tracking.deepsort import DeepSORT
from sklearn.metrics.pairwise import cosine_similarity
from tools.generate_detections import ImageEncoder

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
    encoder_path = os.path.join(os.path.dirname(__file__), 'tracking/models/mars-small128.pb')
    deepsort1 = DeepSORT(encoder_path)
    deepsort2 = DeepSORT(encoder_path)

    video_capture1 = cv2.VideoCapture('test_video/1.mp4')
    video_capture2 = cv2.VideoCapture('test_video/2.mp4')

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

    # ReID matching
    encoder = ImageEncoder(encoder_path)
    feats1 = extract_feature_batch(encoder, [v[0] for v in id_to_crop1.values()])
    feats2 = extract_feature_batch(encoder, [v[0] for v in id_to_crop2.values()])
    ids1 = list(id_to_crop1.keys())
    ids2 = list(id_to_crop2.keys())

    sim = cosine_similarity(np.array([f[1] for f in feats1]), np.array([f[1] for f in feats2]))
    matched_ids = {}
    for i, row in enumerate(sim):
        best_match = np.argmax(row)
        if row[best_match] > 0.8:
            matched_ids[ids2[best_match]] = ids1[i]

    print("Matched IDs:", matched_ids)

    # Save for unified visualization
    np.save("frame_data1.npy", frame_data1, allow_pickle=True)
    np.save("frame_data2.npy", frame_data2, allow_pickle=True)
    np.save("matched_ids.npy", matched_ids)

if __name__ == "__main__":
    main()