import os
import shutil
import torch
import numpy as np
from video_to_frames import extract_frames
from face_embedder import generate_embeddings_from_folder
from facenet_pytorch import InceptionResnetV1

# === 설정 ===
video1_path = "4p-c0.avi"
video2_path = "4p-c1.avi"

frames1_folder = "frames_cam1"
frames2_folder = "frames_cam2"
embeddings1_folder = "embeddings_cam1"
embeddings2_folder = "embeddings_cam2"
matched_faces_dir = "matched_faces"

# === 기존 출력 폴더 정리 ===
for folder in [frames1_folder, frames2_folder, embeddings1_folder, embeddings2_folder, matched_faces_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# === STEP 1: 프레임 추출 ===
print("\n=== STEP 1: Extracting Frames ===")
extract_frames(video1_path, frames1_folder)
extract_frames(video2_path, frames2_folder)

# === STEP 2: 얼굴 임베딩 생성 ===
print("\n=== STEP 2: Generating Face Embeddings ===")
generate_embeddings_from_folder(frames1_folder, embeddings1_folder)
generate_embeddings_from_folder(frames2_folder, embeddings2_folder)

# === STEP 3: 모든 임베딩 간 교차 비교 ===
print("\n=== STEP 3: Cross-matching Embeddings ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matched_pairs = []

embeddings1 = [f for f in os.listdir(embeddings1_folder) if f.endswith('.npy')]
embeddings2 = [f for f in os.listdir(embeddings2_folder) if f.endswith('.npy')]

# 이미 매칭된 항목 재방지용
used_cam2 = set()

for f1 in embeddings1:
    path1 = os.path.join(embeddings1_folder, f1)
    emb1 = np.load(path1)
    if emb1.shape[0] != 512:
        continue
    emb1_tensor = torch.tensor(emb1).to(device)

    for f2 in embeddings2:
        if f2 in used_cam2:
            continue  # 중복 방지

        path2 = os.path.join(embeddings2_folder, f2)
        emb2 = np.load(path2)
        if emb2.shape[0] != 512:
            continue
        emb2_tensor = torch.tensor(emb2).to(device)

        similarity = torch.nn.functional.cosine_similarity(
            emb1_tensor.unsqueeze(0), emb2_tensor.unsqueeze(0), dim=1
        )[0].item()

        if similarity > 0.30:
            print(f"[✅ Same Person] {f1} <==> {f2} | Similarity: {similarity:.4f}")
            matched_pairs.append((f1, f2))
            used_cam2.add(f2)
            break  # 한 사람당 한 명만 매칭

# === STEP 4: 결과 이미지 저장 ===
print("\n=== STEP 4: Saving matched faces ===")
for i, (f1, f2) in enumerate(matched_pairs):
    person_dir = os.path.join(matched_faces_dir, f"person_{i+1}")
    os.makedirs(person_dir, exist_ok=True)

    # .npy 파일 이름 → 원본 프레임 이미지 이름 추정
    jpg1 = f1.replace(".npy", ".jpg")
    jpg2 = f2.replace(".npy", ".jpg")

    img1_path = os.path.join(frames1_folder, jpg1)
    img2_path = os.path.join(frames2_folder, jpg2)

    if os.path.exists(img1_path):
        shutil.copy(img1_path, os.path.join(person_dir, f"cam1_{jpg1}"))
    if os.path.exists(img2_path):
        shutil.copy(img2_path, os.path.join(person_dir, f"cam2_{jpg2}"))

print("\n🎉 DONE! Matched faces saved in:", matched_faces_dir)

