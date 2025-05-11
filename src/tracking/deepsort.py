from tracking.deep_sort.tracker import Tracker
from tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from tools.generate_detections import create_box_encoder
from tracking.deep_sort.detection import Detection

class DeepSORT:
    def __init__(self, model_path):
        self.encoder = create_box_encoder(model_path)
        metric = NearestNeighborDistanceMetric("cosine", 0.4, None)
        self.tracker = Tracker(metric)

    def update(self, frame, bboxes):
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]
        self.tracker.predict()
        self.tracker.update(detections)
        results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            results.append((track.to_tlbr(), track.track_id))
        return results
