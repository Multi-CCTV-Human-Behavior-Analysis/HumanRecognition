import torch
import torchvision.transforms as transforms
from torchvision import models

class FeatureExtractor:
    def __init__(self, model_name='resnet50', pretrained=True):
        self.model = models.__dict__[model_name](pretrained=pretrained)
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image):
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = self.model(image)
        return features.squeeze().numpy()  # Return features as a numpy array

def load_feature_extractor():
    return FeatureExtractor()