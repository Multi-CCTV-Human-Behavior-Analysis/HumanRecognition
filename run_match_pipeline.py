from video_to_frames import extract_frames
from face_embedder import generate_embeddings_from_folder
from compare_embeddings import compare_embeddings
import os

# Paths to your input videos
video1_path = "new video dataset/Falling/istockphoto-1325164835-640_adpp_is - 56.mp4"
video2_path = "new video dataset/Falling/istockphoto-874189204-640_adpp_is.mp4"

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
generate_embeddings_from_folder(frames1_folder, embeddings1_folder)
generate_embeddings_from_folder(frames2_folder, embeddings2_folder)

# STEP 3: Compare embeddings and find matches
print("\n=== STEP 3: Comparing Embeddings ===")
matches = compare_embeddings(embeddings1_folder, embeddings2_folder, threshold=0.75)

# Done!
print("\n🎉 DONE! Thank you!")
