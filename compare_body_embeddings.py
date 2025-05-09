import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compare_body_embeddings(folder1, folder2, threshold=0.7):
    embeddings1 = {
        f: np.load(os.path.join(folder1, f)) for f in os.listdir(folder1) if f.endswith(".npy")
    }
    embeddings2 = {
        f: np.load(os.path.join(folder2, f)) for f in os.listdir(folder2) if f.endswith(".npy")
    }

    print(f"[INFO] Comparing {len(embeddings1)} bodies from Camera 1 to {len(embeddings2)} in Camera 2")

    matches = []

    for name1, emb1 in embeddings1.items():
        for name2, emb2 in embeddings2.items():
            sim = cosine_similarity(emb1, emb2)[0][0]
            if sim > threshold:
                matches.append((name1, name2, sim))

    return matches
