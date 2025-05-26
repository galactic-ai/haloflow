import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.relu = nn.SiLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.fc1(x))
        # x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        # x = self.dropout(x)
        return x

class Classifier(nn.Module):
    def __init__(self, output_dim):
        super(Classifier, self).__init__()
        # self.fc = nn.Linear(32, 32)
        self.fc2  = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, output_dim)
        # self.relu = nn.LeakyReLU(inplace=True)
        self.relu = nn.SiLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x = self.relu(self.fc(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)
        return x

class MMDModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MMDModel, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim)
        self.classifier = Classifier(output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = self.classifier(features)
        return features, outputs
