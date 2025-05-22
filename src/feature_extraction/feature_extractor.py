# reid/feature_extractor.py
import torch
import torchreid
from torchreid.data.transforms import build_transforms
from torchvision import transforms
from PIL import Image
import numpy as np


class TorchReID:
    def __init__(self, model_name='resnet50', model_path=None, gpu_id=0):
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=751,
            loss='softmax',
            pretrained=True
        )
        if model_path:
            torchreid.utils.load_pretrained_weights(self.model, model_path)

        self.model.to(self.device)
        self.model.eval()

        _, self.transform = build_transforms(
            height=256, width=128,
            random_erase=False,
            color_jitter=False,
            color_aug=False
        )

    def extract_features(self, image_paths):
        features = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(img)
            features.append(feat.squeeze(0).cpu().numpy())
        return np.array(features)

    def compute_euclidean_distance(self, feat1, feat2):
        # feat1, feat2 should be numpy arrays
        return np.linalg.norm(feat1 - feat2, axis=1)

    def aggregate_features(self, features):
        return np.mean(features, axis=0)

    def match_tracks(self, features_query_dict, features_gallery_dict, threshold=320):
        matched_ids = {}
        for q_id, q_feats in features_query_dict.items():
            min_dist = float('inf')
            match_id = None
            q_agg = self.aggregate_features(q_feats)
            for g_id, g_feats in features_gallery_dict.items():
                g_agg = self.aggregate_features(g_feats)
                dist = np.linalg.norm(q_agg - g_agg)
                if dist < min_dist:
                    min_dist = dist
                    match_id = g_id
            if min_dist < threshold:
                matched_ids[q_id] = match_id
        return matched_ids
