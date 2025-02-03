"""
Model for the Domain Adversarial Neural Network (DANN) model.
"""
import torch.nn as nn
from torch.autograd import Function

class FeatureExtractor(nn.Module):
    """
    Feature extractor for the DANN model.
    """
    def __init__(self, input_size, hidden_size):
        super(FeatureExtractor, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.feature(x)

class LabelPredictor(nn.Module):
    """
    Label predictor for the DANN model.
    """
    def __init__(self, hidden_size, num_classes):
        super(LabelPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.predictor(x)

class DomainClassifier(nn.Module):
    """
    Domain classifier for the DANN model.
    """
    def __init__(self, hidden_size):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DANN(nn.Module):
    """
    Domain Adversarial Neural Network (DANN) model.
    """
    def __init__(self, input_size, hidden_size, num_classes, alpha):
        super(DANN, self).__init__()
        self.feature_extractor = FeatureExtractor(input_size, hidden_size)
        self.label_predictor = LabelPredictor(hidden_size, num_classes)
        self.domain_classifier = DomainClassifier(hidden_size)
        self.gradient_reversal = GradientReversal(alpha)

    def forward(self, x):
        features = self.feature_extractor(x)
        label_pred = self.label_predictor(features)
        domain_pred = self.domain_classifier(self.gradient_reversal(features))
        return label_pred, domain_pred