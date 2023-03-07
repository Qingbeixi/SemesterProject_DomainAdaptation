import torch.nn as nn

class DAFeatureExtractor(nn.Module):
    def __init__(self):
        super(DAFeatureExtractor, self).__init__()
        
        self.conv1d_1 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=9, stride=1, padding=4) # 50
        self.conv1d_2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=9, stride=1, padding=4)
        self.conv1d_3 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to (batch_size, 20, 50)
        x = self.relu(self.conv1d_1(x)) # 1st convolutional layer
        x = self.relu(self.conv1d_2(x)) # 2nd convolutional layer
        x = self.relu(self.conv1d_3(x)) # 3rd convolutional layer
        x = self.flatten(x) # flatten the output (batch_size, 50)
        features = self.relu(x) # store the features before the RUL regressor
        return features
    
class RULRegressor(nn.Module):
    def __init__(self):
        super(RULRegressor, self).__init__()
        self.fc1 = nn.Linear(50, 50) # fully connected layer 1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1) # fully connected layer 2
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x)) # fully connected layer 1
        x = self.fc2(x) # fully connected layer 2
        x = self.sigmoid(x) # sigmoid activation function to normalize output to (0,1)
        return x

class DA1DCNN(nn.Module):
    def __init__(self):
        super(DA1DCNN, self).__init__()
        self.feature_extractor = DAFeatureExtractor()
        self.rul_regressor = RULRegressor()
        
    def forward(self, x):
        features = self.feature_extractor(x)
        rul = self.rul_regressor(features)
        return rul
