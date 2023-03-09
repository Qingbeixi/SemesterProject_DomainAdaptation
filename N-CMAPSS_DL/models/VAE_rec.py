import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(50*20, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 50*20),
            nn.Sigmoid()
        )

        self.apply(weight_init)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(x.size(0), 50, 20)
        return decoded


class AE(nn.Module):
    def __init__(self, input_dim=20, latent_dim=20, hidden_dim=10):
        super(AE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=0.2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.fc_mu = nn.Linear(latent_dim*12, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim*12, latent_dim)

        # Decoder
        self.fc1 = nn.Linear(latent_dim, hidden_dim*12)
        self.conv4 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.upsample1 = nn.Upsample(scale_factor=25/12, mode='nearest')
        self.dropout4 = nn.Dropout(p=0.2)
        self.conv5 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout5 = nn.Dropout(p=0.2)
        self.conv6 = nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1)

        self.apply(weight_init)

    def encode(self, x):
        x = x.permute(0, 2, 1)  # reshape from (batch_size, 50, 20) to (batch_size, 20, 50) 
        x = self.bn1(F.relu(self.conv1(x))) # (batch_size, 10, 50)
        x = self.dropout1(self.pool1(x)) # (batch_size, 10, 25)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dropout2(self.pool2(x)) # (batch_size, 10, 12)
        x = self.bn3(self.conv3(x)) # (batch_size, 5, 12)
        x = x.view(x.size(0), -1) # (batch_size, 60)
        z = self.fc_mu(x) # (batch_size, 5)
        return z
    
    def decode(self, z):
        z = F.relu(self.fc1(z)) # (batch_size, 5) -> (batch_size, 120)
        z = z.view(z.size(0), -1, 12)  # (batch_size, 10, 12)
        z = self.bn4(F.relu(self.conv4(z))) # (batch_size, 10, 12)
        z = self.dropout4(self.upsample1(z)) # (batch_size, 10, 25)
        z = self.bn5(F.relu(self.conv5(z))) # (batch_size, 10, 25)
        z = self.dropout5(self.upsample2(z))  # (batch_size, 10, 50)
        z = F.relu(self.conv6(z)) # (batch_size, 20, 50)
        x = z.permute(0, 2, 1) # (batch_size, 20, 50)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class AE_Regressor(nn.Module):
    def __init__(self):
        super(AE_Regressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding=1)
        self.fc1 = nn.Linear(32*20, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Special initialization method
        self.apply(weight_init)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the conv layer
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.sigmoid(x)
        x = self.fc4(x)
        return x




class VAE_rec(nn.Module):
    def __init__(self, input_dim=20, latent_dim=32, hidden_dim=10):
        super(VAE_rec, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=0.2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.fc_mu = nn.Linear(latent_dim*12, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim*12, latent_dim)

        # Decoder
        self.fc1 = nn.Linear(latent_dim, hidden_dim*12)
        self.conv4 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.upsample1 = nn.Upsample(scale_factor=25/12, mode='nearest')
        self.dropout4 = nn.Dropout(p=0.2)
        self.conv5 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout5 = nn.Dropout(p=0.2)
        self.conv6 = nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1)

        self.apply(weight_init)

    def encode(self, x):
        x = x.permute(0, 2, 1)  # reshape from (batch_size, 50, 20) to (batch_size, 20, 50) 
        x = self.bn1(F.relu(self.conv1(x))) # (batch_size, 10, 50)
        x = self.dropout1(self.pool1(x)) # (batch_size, 10, 25)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dropout2(self.pool2(x)) # (batch_size, 10, 12)
        x = self.bn3(self.conv3(x)) # (batch_size, 5, 12)
        x = x.view(x.size(0), -1) # (batch_size, 60)
        mu = self.fc_mu(x) # (batch_size, 5)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = F.relu(self.fc1(z)) # (batch_size, 5) -> (batch_size, 120)
        z = z.view(z.size(0), -1, 12)  # (batch_size, 10, 12)
        z = self.bn4(F.relu(self.conv4(z))) # (batch_size, 10, 12)
        z = self.dropout4(self.upsample1(z)) # (batch_size, 10, 25)
        z = self.bn5(F.relu(self.conv5(z))) # (batch_size, 10, 25)
        z = self.dropout5(self.upsample2(z))  # (batch_size, 10, 50)
        z = F.relu(self.conv6(z)) # (batch_size, 20, 50)
        x = z.permute(0, 2, 1) # (batch_size, 20, 50)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar