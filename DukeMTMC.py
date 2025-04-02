from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Input images (update with your image paths)
image_1_path = 'DukeMTMC-reID/DukeMTMC-reID/bounding_box_test/0003_c1_f0044158.jpg'
image_2_path = 'DukeMTMC-reID/DukeMTMC-reID/bounding_box_test/0003_c1_f0046798.jpg'

# Load and preprocess images
img1 = Image.open(image_1_path).convert('RGB')
img2 = Image.open(image_2_path).convert('RGB')

# Detect and align faces
face1 = mtcnn(img1)
face2 = mtcnn(img2)

if face1 is not None and face2 is not None:
    face1 = face1.unsqueeze(0).to(device)
    face2 = face2.unsqueeze(0).to(device)

    # Generate embeddings
    with torch.no_grad():
        emb1 = facenet(face1).cpu().numpy()
        emb2 = facenet(face2).cpu().numpy()

    # Compare embeddings
    similarity = cosine_similarity(emb1, emb2)[0][0]
    print(f"Cosine Similarity: {similarity:.4f}")

    if similarity > 0.7:
        print("✅ SAME PERSON (P1)")
    else:
        print("❌ DIFFERENT PERSON")
else:
    print("Face not detected in one or both images.")
