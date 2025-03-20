"""
Model for the Domain Adversarial Neural Network (DANN) model.
"""

import torch.nn as nn

from . import utils as U


class DANNModel(nn.Module):
    def __init__(self, input_dim):
        super(DANNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module("fc0", nn.Linear(input_dim, 256))
        self.feature.add_module("silu0", nn.SiLU())
        self.feature.add_module("fc1", nn.Linear(256, 128))
        self.feature.add_module("silu1", nn.SiLU())
        self.feature.add_module("fc2", nn.Linear(128, 64))
        self.feature.add_module("silu2", nn.SiLU())
        self.feature.add_module("fc3", nn.Linear(64, 32))
        self.feature.add_module("silu3", nn.SiLU())

        # class classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module("c_fc1", nn.Linear(32, 16))
        self.class_classifier.add_module("c_silu1", nn.SiLU())
        self.class_classifier.add_module("c_fc2", nn.Linear(16, 8))
        self.class_classifier.add_module("c_silu2", nn.SiLU())
        self.class_classifier.add_module("c_fc3", nn.Linear(8, 2))  # sm and hm

        # domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module("d_fc1", nn.Linear(32, 16))
        self.domain_classifier.add_module("d_silu1", nn.SiLU())
        self.domain_classifier.add_module("d_fc2", nn.Linear(16, 8))
        self.domain_classifier.add_module("d_silu2", nn.SiLU())
        self.domain_classifier.add_module("d_fc3", nn.Linear(8, 4))  # 4 domains

    def forward(self, x, alpha):
        x = self.feature(x)
        x_rev = U.GradientReversal.apply(x, alpha)
        label = self.class_classifier(x)
        domain = self.domain_classifier(x_rev)
        return label, domain
