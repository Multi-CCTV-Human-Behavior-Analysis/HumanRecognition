import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compare_body_embeddings(folder1, folder2, threshold=0.7):
    """
    Compares OSNet-based body embeddings from two cameras using cosine similarity.

    Args:
        folder1 (str): Embedding folder for Camera 1.
        folder2 (str): Embedding folder for Camera 2.
        threshold (float): Similarity threshold to declare a match.

    Returns:
        List of matched tuples: (cam1_name, cam2_name, similarity_score)
    """
    embeddings1 = {
        f: np.load(os.path.join(folder1, f))
        for f in os.listdir(folder1) if f.endswith(".npy")
    }
    embeddings2 = {
        f: np.load(os.path.join(folder2, f))
        for f in os.listdir(folder2) if f.endswith(".npy")
    }

    print(f"[INFO] Comparing {len(embeddings1)} body embeddings from Cam1 to {len(embeddings2)} from Cam2")

    matches = []
    for name1, emb1 in embeddings1.items():
        for name2, emb2 in embeddings2.items():
            try:
                sim = cosine_similarity(emb1, emb2)[0][0]
                if sim > threshold:
                    matches.append((name1, name2, round(sim, 4)))
            except Exception:
                continue  # skip invalid or malformed embeddings

    print(f"[✅] {len(matches)} matches found (threshold: {threshold})\n")
    return matches
