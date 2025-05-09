import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def compare_individual_faces(folder1, folder2, threshold=0.75, decision_ratio=0.8):
    """
    Compares grouped face embeddings between two folders (Cam1 and Cam2),
    determines which face IDs likely belong to the same person.

    Args:
        folder1 (str): Face embeddings folder for Camera 1
        folder2 (str): Face embeddings folder for Camera 2
        threshold (float): Cosine similarity threshold to count as a match
        decision_ratio (float): Ratio of frame-to-frame matches required to declare 'same person'
    
    Returns:
        List of match result dictionaries
    """
    def load_embeddings(folder):
        return {
            f: np.load(os.path.join(folder, f))
            for f in os.listdir(folder)
            if f.endswith(".npy")
        }

    def group_by_face(embeddings):
        grouped = defaultdict(list)
        for name, emb in embeddings.items():
            if "_face_" not in name:
                continue
            try:
                face_id = name.split('_face_')[1].split('.')[0]
                grouped[face_id].append((name, emb))
            except IndexError:
                continue
        return grouped

    embeddings1 = load_embeddings(folder1)
    embeddings2 = load_embeddings(folder2)
    grouped1 = group_by_face(embeddings1)
    grouped2 = group_by_face(embeddings2)

    print(f"[INFO] Comparing {len(grouped1)} persons from Cam1 to {len(grouped2)} from Cam2")

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
            decision = "✅ SAME" if match_ratio >= decision_ratio else "❌ DIFF"

            result = {
                "Cam1_Person": id1,
                "Cam2_Person": id2,
                "MatchRatio": round(match_ratio, 4),
                "Status": decision
            }
            all_results.append(result)

    # Print a final summary
    print(f"\n[✅] Comparison complete. Total comparisons: {len(all_results)}")
    matches = [r for r in all_results if r["Status"] == "✅ SAME"]
    print(f"[MATCHES FOUND]: {len(matches)} pairs identified as SAME\n")

    return all_results
