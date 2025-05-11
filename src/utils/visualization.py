import cv2
def draw_bounding_boxes(frame, detections, identities):
    for detection, identity in zip(detections, identities):
        x1, y1, x2, y2 = detection
        color = (0, 255, 0)  # Green color for bounding box
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        frame = cv2.putText(frame, str(identity), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def display_frame(frame, window_name='Frame'):
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

def visualize_tracking_results(video_path, detections, identities):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = draw_bounding_boxes(frame, detections, identities)
        if not display_frame(frame):
            break
    cap.release()
    cv2.destroyAllWindows()