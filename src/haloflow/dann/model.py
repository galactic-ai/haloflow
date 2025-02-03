"""
Model for the Domain Adversarial Neural Network (DANN) model.
"""

import torch.nn as nn
from . import utils as U


class DANN(nn.Module):
    def __init__(self, input_dim, num_domains=4, alpha=1.0):
        super().__init__()
        self.alpha = alpha

        # Feature Extractor (Shared)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )

        # Label Predictor (Task-Specific)
        self.label_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Output: [stellar_mass, halo_mass]
        )

        # Domain Classifier (Adversarial)
        self.domain_classifier = nn.Sequential(
            U.GradientReversalLayer(alpha=self.alpha),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_domains),  # Output: domain logits
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        label_pred = self.label_predictor(features)
        domain_pred = self.domain_classifier(features)
        return label_pred, domain_pred

    def build_from_config(cls, config):
        """Optional: For hyperparameter flexibility"""
        return cls(input_dim=config["input_dim"], num_domains=config["num_domains"])
