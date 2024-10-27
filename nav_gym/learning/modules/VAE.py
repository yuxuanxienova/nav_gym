import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VariationalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc31(h2)
        log_var = self.fc32(h2)
        return mu, log_var

class VariationalDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super(VariationalDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, z):
        h1 = self.relu(self.fc1(z))
        h2 = self.relu(self.fc2(h1))
        return self.fc3(h2)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = VariationalDecoder(latent_dim, input_dim, hidden_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD