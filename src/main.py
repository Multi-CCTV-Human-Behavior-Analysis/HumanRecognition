import cv2
from detection.yolo_model import load_yolo_model
import random

def main():
    # Load YOLO model
    yolo_model = load_yolo_model('yolov8n.pt')  # Replace with the correct path to your YOLOv8 weights

    # Open video files
    video_capture1 = cv2.VideoCapture('test_video/1.mp4')  # First video input
    video_capture2 = cv2.VideoCapture('test_video/2.mp4')  # Second video input

    # Get the frame dimensions for the video writers
    ret1, frame1 = video_capture1.read()
    ret2, frame2 = video_capture2.read()
    if not ret1 or not ret2:
        print("Error: Unable to read video files.")
        return

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter('output1.mp4', fourcc, 30.0, (frame1.shape[1], frame1.shape[0]))
    out2 = cv2.VideoWriter('output2.mp4', fourcc, 30.0, (frame2.shape[1], frame2.shape[0]))

    # Initialize a counter for unique IDs
    person_id_counter = 1
    person_ids = {}  # Dictionary to store IDs for detected bounding boxes
    person_colors = {}  # Dictionary to store colors for each person ID

    while True:
        # Read frames from both videos
        ret1, frame1 = video_capture1.read()
        ret2, frame2 = video_capture2.read()

        # Break the loop if either video ends
        if not ret1 or not ret2:
            break

        # Detect people in the first video
        detections1 = yolo_model.detect_people(frame1)
        for i, (x, y, w, h) in enumerate(detections1):
            # Assign a unique ID to each detection
            if i not in person_ids:
                person_ids[i] = f"P{person_id_counter}"
                person_colors[person_ids[i]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                person_id_counter += 1

            # Draw bounding box and ID
            person_id = person_ids[i]
            color = person_colors[person_id]
            cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame1, person_id, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Detect people in the second video
        detections2 = yolo_model.detect_people(frame2)
        for i, (x, y, w, h) in enumerate(detections2):
            # Assign a unique ID to each detection
            if i not in person_ids:
                person_ids[i] = f"P{person_id_counter}"
                person_colors[person_ids[i]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                person_id_counter += 1

            # Draw bounding box and ID
            person_id = person_ids[i]
            color = person_colors[person_id]
            cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame2, person_id, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the processed frames to the output videos
        out1.write(frame1)
        out2.write(frame2)

        # Display both frames (optional, can be removed in headless environments)
        cv2.imshow("YOLOv8 Detection - Video 1", frame1)
        cv2.imshow("YOLOv8 Detection - Video 2", frame2)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture1.release()
    video_capture2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()