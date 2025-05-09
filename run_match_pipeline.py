import os
from video_to_frames import extract_frames
from face_embedder import generate_multi_embeddings
from compare_embeddings import compare_individual_faces
from crop_persons_yolo import crop_persons_from_frames
from body_embedder import generate_body_embeddings
from compare_body_embeddings import compare_body_embeddings

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

# ==== Clean output folders ====
for folder in [frames1_folder, frames2_folder, face_embeds1, face_embeds2,
               person_crops1, person_crops2, body_embeds1, body_embeds2]:
    os.makedirs(folder, exist_ok=True)

# ==== STEP 1: Extract frames ====
print("\n=== STEP 1: Extracting Frames ===")
extract_frames(video1_path, frames1_folder)
extract_frames(video2_path, frames2_folder)

# ==== STEP 2: Generate face embeddings ====
print("\n=== STEP 2: Generating Face Embeddings ===")
generate_multi_embeddings(frames1_folder, face_embeds1)
generate_multi_embeddings(frames2_folder, face_embeds2)

# ==== STEP 3: Compare face embeddings ====
print("\n=== STEP 3: Comparing Face Embeddings ===")
compare_individual_faces(face_embeds1, face_embeds2, threshold=0.75, decision_ratio=0.8)

# ==== STEP 4: Crop people using YOLOv8 ====
print("\n=== STEP 4: Cropping Persons using YOLOv8 ===")
crop_persons_from_frames(frames1_folder, person_crops1)
crop_persons_from_frames(frames2_folder, person_crops2)

# ==== STEP 5: Generate body embeddings (OSNet) ====
print("\n=== STEP 5: Generating Body Embeddings ===")
generate_body_embeddings(person_crops1, body_embeds1)
generate_body_embeddings(person_crops2, body_embeds2)

# ==== STEP 6: Compare body embeddings ====
print("\n=== STEP 6: Comparing Body Embeddings ===")
compare_body_embeddings(body_embeds1, body_embeds2, threshold=0.75)

# ==== Done ====
print("\n🎉 DONE! Both face-based and body-based comparison completed successfully.")
