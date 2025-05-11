from ultralytics import YOLO

class YOLOModel:
    def __init__(self, weights_path, conf_threshold=0.5):
        # Load the YOLOv8 model
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect_people(self, frame):
        # Perform inference
        results = self.model.predict(source=frame, conf=self.conf_threshold, save=False, save_txt=False)

        # Extract bounding boxes for detected people
        detected_boxes = []
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:  # Assuming class 0 corresponds to 'person'
                    # Extract coordinates from the tensor and convert to integers
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    detected_boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to [x, y, w, h]

        return detected_boxes

def load_yolo_model(weights_path):
    return YOLOModel(weights_path)