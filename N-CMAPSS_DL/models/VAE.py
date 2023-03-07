"created by qingjun, variational auto encoder"

import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F

# class VAE(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_dim):
#         super(VAE, self).__init__()
#         self.input_dim = input_dim
#         # Encoder layers
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc21 = nn.Linear(hidden_dim, latent_dim)
#         self.fc22 = nn.Linear(hidden_dim, latent_dim)

#         # Decoder layers
#         self.fc3 = nn.Linear(latent_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, input_dim)

#     def encode(self, x):
#         h = F.relu(self.fc1(x))
#         mu = self.fc21(h)
#         logvar = self.fc22(h)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h = F.relu(self.fc3(z))
#         y = self.fc4(h)
#         return y

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, self.input_dim))
#         z = self.reparameterize(mu, logvar)
#         y = self.decode(z)
#         return y, mu, logvar
    
# class VAERegressor(VAE):
#     def __init__(self, input_dim, latent_dim, hidden_dim):
#         super(VAERegressor, self).__init__(input_dim, latent_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, 1)

#     def decode(self, z):
#         h = F.relu(self.fc3(z))
#         y = self.fc4(h)
#         return y


# class VAE_1D(nn.Module):
#     def __init__(self, input_dim=20, latent_dim=5, hidden_dim=10):
#         super(VAE_1D, self).__init__()

#         # Encoder
#         self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
#         self.fc_mu = nn.Linear(latent_dim*50, latent_dim)
#         self.fc_logvar = nn.Linear(latent_dim*50, latent_dim)

#         # Decoder
#         self.fc1 = nn.Linear(latent_dim, hidden_dim*25)
#         self.conv4 = nn.ConvTranspose1d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1)

#     def encode(self, x):
#         x = x.permute(0, 2, 1)  # reshape from (batch_size, 50, 20) to (batch_size, 20, 50)
#         x = F.relu(self.conv1(x))# (batch_size, 10, 50)
#         x = F.relu(self.conv2(x)) # (batch_size, 10, 50)
#         x = self.conv3(x) # (batch_size, 5, 50)
#         x = x.view(x.size(0), -1)# (batch_size, 250)
#         mu = self.fc_mu(x) # (batch_size, 250) => (batch_size, 5)
#         logvar = self.fc_logvar(x)
#         return mu, logvar

#     def decode(self, z):
#         z = F.relu(self.fc1(z))# (batch_size, 5) => (batch_size, 250)
#         z = z.view(z.size(0), -1, 50) #(batch_size, 5,50)
#         z = F.relu(self.conv4(z)) #(batch_size, 10,50)
#         x = torch.sigmoid(self.conv5(z)) #(batch_size, 20,50)
#         x = x.permute(0, 2, 1)  # reshape from (batch_size, 20, 50) to (batch_size, 50, 20)
#         return x

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         z = mu + eps * std
#         return z

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         x_hat = self.decode(z)
#         return x_hat, mu, logvar
class VAE_1D(nn.Module):
    def __init__(self, input_dim=20, latent_dim=5, hidden_dim=10):
        super(VAE_1D, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.fc_mu = nn.Linear(latent_dim*50, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim*50, latent_dim)

        # Decoder
        self.fc1 = nn.Linear(latent_dim, hidden_dim*25)
        self.conv4 = nn.ConvTranspose1d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.conv5 = nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = x.permute(0, 2, 1)  # reshape from (batch_size, 50, 20) to (batch_size, 20, 50)
        x = self.bn1(F.relu(self.conv1(x)))# (batch_size, 10, 50)
        x = self.bn2(F.relu(self.conv2(x))) # (batch_size, 10, 50)
        x = self.bn3(self.conv3(x)) # (batch_size, 5, 50)
        x = x.view(x.size(0), -1)# (batch_size, 250)
        mu = self.fc_mu(x) # (batch_size, 250) => (batch_size, 5)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = F.relu(self.fc1(z))# (batch_size, 5) => (batch_size, 250)
        z = z.view(z.size(0), -1, 50) #(batch_size, 5,50)
        z = self.bn4(F.relu(self.conv4(z))) #(batch_size, 10,50)
        x = torch.sigmoid(self.conv5(z)) #(batch_size, 20,50)
        x = x.permute(0, 2, 1)  # reshape from (batch_size, 20, 50) to (batch_size, 50, 20)
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
    
class VAERegressor(VAE_1D):
    def __init__(self, input_dim=20, latent_dim=5, hidden_dim=10):
        super(VAERegressor, self).__init__(input_dim, latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim*25, 1)
        self.sigmoid = nn.Sigmoid()

    def decode(self, z):
        h = F.relu(self.fc1(z)) # (batch_size, 250)
        y = self.sigmoid(self.fc4(h))
        return y