"created by qingjun, variational auto encoder"

import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc21(h)
        logvar = self.fc22(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        y = self.fc4(h)
        return y

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return y, mu, logvar
    
class VAERegressor(VAE):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAERegressor, self).__init__(input_dim, latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        y = self.fc4(h)
        return y