import os
from video_to_frames import extract_frames
from face_embedder import generate_multi_embeddings
from crop_persons_yolo import crop_persons_from_frames
from body_embedder import generate_body_embeddings
from hybrid_compare import hybrid_compare
from track_and_crop import track_and_crop
# from hybrid_compare import export_to_csv  # Optional: if you want to export matches to CSV

# ==== Video Input Paths ====
video1_path = "test_video/view-HC2.mp4"
video2_path = "test_video/view-HC3.mp4"

# ==== Output Folders ====
frames1_folder = "frames_cam1"
frames2_folder = "frames_cam2"
face_embeds1 = "embeddings_cam1"
face_embeds2 = "embeddings_cam2"
person_crops1 = "person_crops_cam1"
person_crops2 = "person_crops_cam2"
body_embeds1 = "body_embeddings_cam1"
body_embeds2 = "body_embeddings_cam2"
tracked_crops1 = "tracked_crops_cam1"
tracked_crops2 = "tracked_crops_cam2"

# ==== Create Folders if Needed ====
for folder in [frames1_folder, frames2_folder, face_embeds1, face_embeds2,
               person_crops1, person_crops2, body_embeds1, body_embeds2,
               tracked_crops1, tracked_crops2]:
    os.makedirs(folder, exist_ok=True)

# ==== STEP 1: Extract Frames ====
print("\n=== STEP 1: Extracting Frames ===")
extract_frames(video1_path, frames1_folder)
extract_frames(video2_path, frames2_folder)

# ==== STEP 2: FaceNet Embeddings ====
print("\n=== STEP 2: Generating Face Embeddings (FaceNet) ===")
generate_multi_embeddings(frames1_folder, face_embeds1)
generate_multi_embeddings(frames2_folder, face_embeds2)

# ==== STEP 3: YOLOv8 Person Cropping ====
print("\n=== STEP 3: Cropping Persons using YOLOv8 ===")
crop_persons_from_frames(frames1_folder, person_crops1)
crop_persons_from_frames(frames2_folder, person_crops2)

# ==== STEP 4: OSNet Body Embeddings ====
print("\n=== STEP 4: Generating Body Embeddings (OSNet) ===")
generate_body_embeddings(person_crops1, body_embeds1)
generate_body_embeddings(person_crops2, body_embeds2)

# ==== STEP 5: Hybrid Comparison (Face + Body) ====
print("\n=== STEP 5: Hybrid Person Matching (Face + Body) ===")
matches = hybrid_compare(
    face_dir1=face_embeds1,
    face_dir2=face_embeds2,
    body_dir1=body_embeds1,
    body_dir2=body_embeds2,
    threshold=0.75,
    w_face=0.6,
    w_body=0.4
)

# Optional: Export to CSV
# export_to_csv(matches, "results/hybrid_matches.csv")

# ==== STEP 6: DeepSORT Tracking & Cropping ====
print("\n=== STEP 6: Tracking and Cropping Persons (DeepSORT) ===")
track_and_crop(video1_path, tracked_crops1)
track_and_crop(video2_path, tracked_crops2)

# ==== DONE ====
print("\n🎉 DONE! Full hybrid person re-identification pipeline executed successfully.")
