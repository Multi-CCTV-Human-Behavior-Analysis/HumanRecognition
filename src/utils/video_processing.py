import cv2

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def display_frame(frame, window_name='Video Frame'):
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)

def save_video(output_path, frames, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def handle_video_stream(video_source):
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        display_frame(frame)
    
    cap.release()
    cv2.destroyAllWindows()