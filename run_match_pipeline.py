import os
from video_to_frames import extract_frames
from face_embedder import generate_multi_embeddings
from crop_persons_yolo import crop_persons_from_frames
from body_embedder import generate_body_embeddings
from hybrid_compare import hybrid_compare
from track_and_crop import track_and_crop

# ==== Paths ====
video1_path = "test_video/view-HC2.mp4"
video2_path = "test_video/view-HC3.mp4"

frames1_folder = "frames_cam1"
frames2_folder = "frames_cam2"
face_embeds1 = "embeddings_cam1"
face_embeds2 = "embeddings_cam2"
person_crops1 = "person_crops_cam1"
person_crops2 = "person_crops_cam2"
body_embeds1 = "body_embeddings_cam1"
body_embeds2 = "body_embeddings_cam2"

# ==== Clean/Create output folders ====
for folder in [frames1_folder, frames2_folder, face_embeds1, face_embeds2,
               person_crops1, person_crops2, body_embeds1, body_embeds2]:
    os.makedirs(folder, exist_ok=True)

# ==== STEP 1: Extract video frames ====
print("\n=== STEP 1: Extracting Frames ===")
extract_frames(video1_path, frames1_folder)
extract_frames(video2_path, frames2_folder)

# ==== STEP 2: Generate FaceNet embeddings ====
print("\n=== STEP 2: Generating Face Embeddings (FaceNet) ===")
generate_multi_embeddings(frames1_folder, face_embeds1)
generate_multi_embeddings(frames2_folder, face_embeds2)

# ==== STEP 3: Crop person images using YOLOv8 ====
print("\n=== STEP 3: Cropping Persons using YOLOv8 ===")
crop_persons_from_frames(frames1_folder, person_crops1)
crop_persons_from_frames(frames2_folder, person_crops2)

# ==== STEP 4: Generate OSNet body embeddings ====
print("\n=== STEP 4: Generating Body Embeddings (OSNet) ===")
generate_body_embeddings(person_crops1, body_embeds1)
generate_body_embeddings(person_crops2, body_embeds2)

# ==== STEP 5: Hybrid comparison (Face + Body combined) ====
print("\n=== STEP 5: Hybrid Person Matching (Face + Body) ===")
hybrid_compare(
    face_dir1=face_embeds1,
    face_dir2=face_embeds2,
    body_dir1=body_embeds1,
    body_dir2=body_embeds2,
    threshold=0.75,
    w_face=0.6,
    w_body=0.4
)

# ==== STEP 6: Track and crop persons in video ====
print("\n=== STEP 6: Tracking and Cropping Persons in Video ===")
track_and_crop("test_video/view-HC2.mp4", "tracked_crops_cam1")
track_and_crop("test_video/view-HC3.mp4", "tracked_crops_cam2")

# ==== Done ====
print("\n🎉 DONE! Hybrid re-identification completed successfully.")
