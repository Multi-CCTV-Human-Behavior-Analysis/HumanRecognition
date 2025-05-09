import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def generate_multi_embeddings(folder_path, output_path):
    """
    Detects all faces in images from `folder_path`, generates embeddings using FaceNet,
    and saves them to `output_path` as .npy files.
    
    Args:
        folder_path (str): Folder containing input frames.
        output_path (str): Folder to save face embeddings.
    """
    os.makedirs(output_path, exist_ok=True)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    total_faces = 0

    for img_name in sorted(image_files):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            faces = mtcnn(img)

            if faces is not None:
                for idx, face in enumerate(faces):
                    face_tensor = face.unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = facenet(face_tensor).cpu().numpy()
                    base_name = os.path.splitext(img_name)[0]
                    save_name = f"{base_name}_face_{idx}.npy"
                    np.save(os.path.join(output_path, save_name), embedding)
                    total_faces += 1

        except Exception as e:
            print(f"[WARNING] Failed on {img_name}: {e}")

    print(f"[✅] {total_faces} face embeddings saved to '{output_path}'")
