import os
import cv2
import torch
import numpy as np
from torchreid.utils import FeatureExtractor

# Load the OSNet model
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def generate_body_embeddings(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    for img_name in sorted(files):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARNING] Skipping {img_name} (unable to read)")
            continue

        img_resized = cv2.resize(img, (128, 256))  # (W, H)
        features = extractor(img_resized)
        embedding = features.cpu().numpy()

        save_path = os.path.join(output_folder, img_name.replace('.jpg', '.npy').replace('.png', '.npy'))
        np.save(save_path, embedding)
        print(f"[INFO] Saved body embedding for {img_name}")
