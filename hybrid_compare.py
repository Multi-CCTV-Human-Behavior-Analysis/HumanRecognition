import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def hybrid_compare(face_dir1, face_dir2, body_dir1, body_dir2, threshold=0.75, w_face=0.6, w_body=0.4):
    """
    Compares person embeddings using a weighted combination of face and body similarity.

    Args:
        face_dir1 (str): Folder of face embeddings (Camera 1)
        face_dir2 (str): Folder of face embeddings (Camera 2)
        body_dir1 (str): Folder of body embeddings (Camera 1)
        body_dir2 (str): Folder of body embeddings (Camera 2)
        threshold (float): Match threshold for combined similarity
        w_face (float): Weight for face similarity
        w_body (float): Weight for body similarity

    Returns:
        List of dicts with match results
    """
    face_emb1 = {
        f.split('.')[0]: np.load(os.path.join(face_dir1, f)) 
        for f in os.listdir(face_dir1) if f.endswith(".npy")
    }
    face_emb2 = {
        f.split('.')[0]: np.load(os.path.join(face_dir2, f)) 
        for f in os.listdir(face_dir2) if f.endswith(".npy")
    }
    body_emb1 = {
        f.split('.')[0]: np.load(os.path.join(body_dir1, f)) 
        for f in os.listdir(body_dir1) if f.endswith(".npy")
    }
    body_emb2 = {
        f.split('.')[0]: np.load(os.path.join(body_dir2, f)) 
        for f in os.listdir(body_dir2) if f.endswith(".npy")
    }

    print(f"\n[INFO] Hybrid matching — comparing {len(face_emb1)} face IDs and {len(body_emb1)} body IDs from Cam1 to Cam2...\n")

    matches = []

    for id1 in face_emb1:
        for id2 in face_emb2:
            # Face similarity
            face_sim = cosine_similarity(face_emb1[id1], face_emb2[id2])[0][0]

            # Body ID format: frame_3_face_0 → frame_3_p0
            body1_id = id1.replace("frame_", "").replace("face_", "p")
            body2_id = id2.replace("frame_", "").replace("face_", "p")

            # Body similarity
            body_sim = 0.0
            if body1_id in body_emb1 and body2_id in body_emb2:
                body_sim = cosine_similarity(body_emb1[body1_id], body_emb2[body2_id])[0][0]

            # Final decision
            final_score = w_face * face_sim + w_body * body_sim
            decision = "✅ SAME" if final_score >= threshold else "❌ DIFF"

            matches.append({
                "Person1": id1,
                "Person2": id2,
                "FaceSim": round(face_sim, 4),
                "BodySim": round(body_sim, 4),
                "FinalScore": round(final_score, 4),
                "Decision": decision
            })

    same_count = sum(1 for m in matches if m["Decision"] == "✅ SAME")
    print(f"[✅] Hybrid comparison complete — {same_count} matches found over {len(matches)} pairs.\n")

    return matches
