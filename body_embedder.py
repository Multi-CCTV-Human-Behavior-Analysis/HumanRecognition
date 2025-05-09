import os
import cv2
import torch
import numpy as np
from torchreid.utils import FeatureExtractor

# Load the OSNet model for person Re-ID
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def generate_body_embeddings(image_folder, output_folder):
    """
    Generates OSNet body embeddings from cropped person images.
    
    Args:
        image_folder (str): Folder with cropped person images (128x256 recommended)
        output_folder (str): Folder to save .npy embeddings
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))]

    saved = 0
    for img_name in sorted(image_files):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        try:
            resized = cv2.resize(img, (128, 256))  # (W, H)
            features = extractor(resized)
            embedding = features.cpu().numpy()

            embed_name = os.path.splitext(img_name)[0] + ".npy"
            np.save(os.path.join(output_folder, embed_name), embedding)
            saved += 1
        except Exception as e:
            print(f"[WARNING] Failed to process {img_name}: {e}")

    print(f"[✅] Saved {saved} body embeddings to '{output_folder}'")
