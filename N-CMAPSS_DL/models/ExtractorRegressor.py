import torch
import torch.nn as nn
import torch.optim as optim


class FeatureExtractor_simple(nn.Module):
    """A feature extractor that uses 1D convolution to extract features from time series data."""
    
    def __init__(self):
        super(FeatureExtractor_simple, self).__init__()
        
        self.conv1d_1 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense_layer = nn.Linear(50, 50)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to (batch_size, 20, 50)
        x = self.relu(self.conv1d_1(x))  # Apply 1D convolution
        x = self.relu(self.conv1d_2(x))
        x = self.relu(self.conv1d_3(x))
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, 50, 1)
        x = self.flatten(x)  # Flatten the tensor (batch_size,50)
        x = self.dense_layer(x)  # Apply the dense layer
        x = self.relu(x)
        return x  # (batch_size,50)



class FeatureExtractor(nn.Module):
    """A feature extractor that uses 1D convolution to extract features from time series data."""
    
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.conv1d_1 = nn.Conv1d(in_channels=20, out_channels=32, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm1d(32)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm1d(64)
        self.conv1d_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm1d(128)
        self.conv1d_4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm1d(256)
        self.conv1d_5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn_5 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense_layer_1 = nn.Linear(512, 256)
        self.bn_dense_1 = nn.BatchNorm1d(256)
        self.dense_layer_2 = nn.Linear(256, 128)
        self.bn_dense_2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to (batch_size, 20, 50)
        x = self.relu(self.bn_1(self.conv1d_1(x)))  # Apply 1D convolution and batch normalization
        x = self.pool(x)  # Apply max pooling
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.bn_2(self.conv1d_2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.bn_3(self.conv1d_3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.bn_4(self.conv1d_4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.bn_5(self.conv1d_5(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, 50, 512)
        x = self.flatten(x)  # Flatten the tensor (batch_size, 50*512)
        x = self.relu(self.bn_dense_1(self.dense_layer_1(x)))  # Apply the dense layer and batch normalization
        x = self.dropout(x)
        x = self.relu(self.bn_dense_2(self.dense_layer_2(x)))
        x = self.dropout(x)
        return x  # (batch_size, 128)
    

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
        self.regressor = Regressor(128, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x