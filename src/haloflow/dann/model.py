"""
Model for the Domain Adversarial Neural Network (DANN) model.
"""

import torch.nn as nn
from . import utils as U


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, layers=[128, 64], dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, layers[0])])
        for i in range(len(layers) - 1):
            self.layers.extend(
                [
                    nn.Linear(layers[i], layers[i + 1]),
                    nn.BatchNorm1d(layers[i + 1]),
                    nn.Dropout(dropout),
                ]
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                x = layer(x)
            else:
                x = self.relu(layer(x))
        return x


class LabelPredictor(nn.Module):
    def __init__(self, label_layers=[64, 32], output_dim=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(label_layers) - 1):
            self.layers.extend(
                [
                    nn.Linear(label_layers[i], label_layers[i + 1]),
                    nn.BatchNorm1d(label_layers[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ]
            )
        self.output = nn.Linear(label_layers[-1], output_dim)
        self.dropout = dropout
        # Output: [stellar_mass, halo_mass]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class DomainClassifier(nn.Module):
    def __init__(self, domain_layers=[64, 32], num_domains=4, alpha=1.0):
        super().__init__()
        self.grl = U.GradientReversalLayer(alpha=alpha)
        self.layers = nn.ModuleList()
        for i in range(len(domain_layers) - 1):
            self.layers.extend(
                [
                    nn.Linear(domain_layers[i], domain_layers[i + 1]),
                    nn.ReLU(),
                    # nn.Dropout(0.5)
                ]
            )
        self.output = nn.Linear(domain_layers[-1], num_domains)

    def forward(self, x):
        x = self.grl(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class DANN(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_layers=[128, 64],
        label_layers=[64, 32],
        output_dim=2,
        domain_layers=[64, 32],
        num_domains=4,
        alpha=1.0,
    ):
        super().__init__()
        self.alpha = alpha

        # Feature Extractor (Shared)
        self.feature_extractor = FeatureExtractor(input_dim, feature_layers)

        # Label Predictor (Task-Specific)
        self.label_predictor = LabelPredictor(label_layers, output_dim)

        # Domain Classifier (Adversarial)
        self.domain_classifier = DomainClassifier(domain_layers, num_domains, alpha)

    def forward(self, x):
        features = self.feature_extractor(x)
        label_pred = self.label_predictor(features)
        domain_pred = self.domain_classifier(features)
        return label_pred, domain_pred

def build_from_config(cls, config):
    """Optional: For hyperparameter flexibility"""
    return cls(
        input_dim=config["input_dim"],
        num_domains=config["num_domains"],
        feature_layers=config["feature_layers"],
        label_layers=config["label_layers"],
        domain_layers=config["domain_layers"],
        alpha=config["alpha"],
    )
