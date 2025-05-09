import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def hybrid_compare(face_dir1, face_dir2, body_dir1, body_dir2, threshold=0.75, w_face=0.6, w_body=0.4):
    face_emb1 = {f.split('.')[0]: np.load(os.path.join(face_dir1, f)) for f in os.listdir(face_dir1) if f.endswith(".npy")}
    face_emb2 = {f.split('.')[0]: np.load(os.path.join(face_dir2, f)) for f in os.listdir(face_dir2) if f.endswith(".npy")}
    body_emb1 = {f.split('.')[0]: np.load(os.path.join(body_dir1, f)) for f in os.listdir(body_dir1) if f.endswith(".npy")}
    body_emb2 = {f.split('.')[0]: np.load(os.path.join(body_dir2, f)) for f in os.listdir(body_dir2) if f.endswith(".npy")}

    print(f"\n[INFO] Comparing {len(face_emb1)} faces and {len(body_emb1)} bodies from Cam1 to Cam2...\n")
    
    matches = []

    # Match face and body embedding files with similar IDs
    for id1 in face_emb1:
        for id2 in face_emb2:
            # Compute face similarity
            face_sim = cosine_similarity(face_emb1[id1], face_emb2[id2])[0][0] if id1 in face_emb1 and id2 in face_emb2 else 0.0
            
            # Check if corresponding body crops exist
            body1_id = id1.replace("frame_", "").replace("face_", "p")  # e.g., frame_3_face_0 → frame_3_p0
            body2_id = id2.replace("frame_", "").replace("face_", "p")

            body_sim = 0.0
            if body1_id in body_emb1 and body2_id in body_emb2:
                body_sim = cosine_similarity(body_emb1[body1_id], body_emb2[body2_id])[0][0]

            final_score = w_face * face_sim + w_body * body_sim
            decision = "✅ SAME" if final_score > threshold else "❌ DIFF"

            matches.append({
                "Person1": id1,
                "Person2": id2,
                "FaceSim": face_sim,
                "BodySim": body_sim,
                "FinalScore": final_score,
                "Decision": decision
            })
            
            print(f"{id1} ⇄ {id2} | Face: {face_sim:.2f} | Body: {body_sim:.2f} | Final: {final_score:.2f} → {decision}")

    return matches
