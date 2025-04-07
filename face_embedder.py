import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

# Initialize face detector (MTCNN) and FaceNet model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to process all images in a folder
def generate_embeddings_from_folder(folder_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)

            if face is not None:
                face = face.unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = facenet(face).cpu().numpy()
                    save_name = os.path.splitext(img_name)[0] + ".npy"
                    np.save(os.path.join(output_path, save_name), embedding)
                    print(f"[INFO] Saved embedding for {img_name}")
            else:
                print(f"[WARNING] No face detected in {img_name}")

        except Exception as e:
            print(f"[ERROR] Failed processing {img_name}: {e}")
