from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

class ReIDModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = torch.load(model_path)
        model.eval()
        return model

    def extract_features(self, images):
        with torch.no_grad():
            features = self.model(images)
        return features

    def compute_similarity(self, features1, features2):
        features1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
        features2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
        similarity = cosine_similarity(features1, features2)
        return similarity

    def re_identify(self, features_query, features_gallery):
        similarity = self.compute_similarity(features_query, features_gallery)
        return np.argmax(similarity, axis=1)  # Return indices of the most similar gallery features
    
    def match(self, features_query, features_gallery):
        similarity = self.compute_similarity(features_query, features_gallery)
        matches = np.argmax(similarity, axis=1)
        return matches
