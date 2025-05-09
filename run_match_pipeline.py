from face_embedder import generate_multi_embeddings
from compare_embeddings import compare_individual_faces
from video_to_frames import extract_frames
import os

# Paths to your input videos
video1_path = "test_video/view-HC2.mp4"
video2_path = "test_video/view-HC3.mp4"

# Output folders
frames1_folder = "frames_cam1"
frames2_folder = "frames_cam2"
embeddings1_folder = "embeddings_cam1"
embeddings2_folder = "embeddings_cam2"

# Clean previous outputs as the folder will clash
for folder in [frames1_folder, frames2_folder, embeddings1_folder, embeddings2_folder]:
    os.makedirs(folder, exist_ok=True)

# STEP 1: Convert videos to frames
print("\n=== STEP 1: Extracting Frames ===")
extract_frames(video1_path, frames1_folder)
extract_frames(video2_path, frames2_folder)

# STEP 2: Generate face embeddings
print("\n=== STEP 2: Generating Face Embeddings ===")
generate_multi_embeddings(frames1_folder, embeddings1_folder)
generate_multi_embeddings(frames2_folder, embeddings2_folder)

# STEP 3: Compare embeddings and find matches
print("\n=== STEP 3: Comparing Embeddings ===")
compare_individual_faces(embeddings1_folder, embeddings2_folder, threshold=0.75, decision_ratio=0.8)

# Done!
print("\n🎉 DONE! Thank you!")
