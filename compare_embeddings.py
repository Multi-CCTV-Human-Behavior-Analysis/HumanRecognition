import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def compare_individual_faces(folder1, folder2, threshold=0.75, decision_ratio=0.8):
    # Load embeddings
    embeddings1 = {
        f: np.load(os.path.join(folder1, f))
        for f in os.listdir(folder1) if f.endswith(".npy")
    }
    embeddings2 = {
        f: np.load(os.path.join(folder2, f))
        for f in os.listdir(folder2) if f.endswith(".npy")
    }

    # Group by face ID pattern
    def group_by_face(embeddings):
        grouped = defaultdict(list)
        for name, emb in embeddings.items():
            face_id = name.split('_face_')[1].split('.')[0]  # e.g. "0"
            grouped[face_id].append((name, emb))
        return grouped

    grouped1 = group_by_face(embeddings1)
    grouped2 = group_by_face(embeddings2)

    print(f"[INFO] Matching {len(grouped1)} persons from Cam1 to {len(grouped2)} persons in Cam2")

    all_results = []

    for id1, faces1 in grouped1.items():
        for id2, faces2 in grouped2.items():
            match_count = 0
            total = 0
            for _, emb1 in faces1:
                for _, emb2 in faces2:
                    sim = cosine_similarity(emb1, emb2)[0][0]
                    total += 1
                    if sim > threshold:
                        match_count += 1
            match_ratio = match_count / total if total else 0
            result = {
                "Cam1_Person": id1,
                "Cam2_Person": id2,
                "MatchRatio": match_ratio,
                "Status": "✅ SAME" if match_ratio >= decision_ratio else "❌ DIFF"
            }
            all_results.append(result)

    return all_results
