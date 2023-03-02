import torch
import torch.nn as nn
import torch.optim as optim


# Define the feature extractor model
class FeatureExtractor(nn.Module):
    """receive input (batch_size,window_size,feature_size) and extractor feature with 1D convolution, transform it to (batch_size,window_size)

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.conv1d_1 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(50, 50)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to (batch_size, 20, 50)
        x = self.relu(self.conv1d_1(x))  # Apply 1D convolution
        x = self.relu(self.conv1d_2(x))
        x = self.relu(self.conv1d_3(x))
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, 50, 1)
        x = self.flatten(x)  # Flatten the tensor (batch_size,50)
        x = self.fc(x)  # Apply the dense layer
        x = self.relu(x)
        return x # (batch_size,50)

# Define the regressor model
class Regressor(nn.Module):
    def __init__(self,feature_size, output_size):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(feature_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        return x

# Define the full model
class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.regressor = Regressor(50, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x