import torch
import torch.nn as nn
import torch.nn.functional as F
from nav_gym.learning.modules.transformer import TransformerEncoder, TransformerDecoder
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
    def __init__(self, latent_dim, hidden_dim, output_dim):
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
        self.decoder = VariationalDecoder(latent_dim, hidden_dim, input_dim)

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
    def save(self, path, optimizer=None):
        """
        Save the model state_dict and optionally the optimizer state_dict.
        """
        if optimizer:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
        else:
            torch.save(self.state_dict(), path)

    def load(self, path, optimizer=None):
        """
        Load the model state_dict and optionally the optimizer state_dict.
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
class tVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_input_tokens):
        super(tVAE, self).__init__()
        token_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, token_dim)
        # Transformer Encoder
        encoder_args = {
            "depth": 1,
            "n_heads": 2,
            "entity_dim": token_dim,
            "num_input_tokens": num_input_tokens,
            "out_dim": token_dim,  # Encoder outputs embeddings of size embed_dim
        }
        self.encoder = TransformerEncoder(encoder_args)
        self.fc31 = nn.Linear(token_dim, latent_dim)
        self.fc32 = nn.Linear(token_dim, latent_dim)

        self.decoder = VariationalDecoder(latent_dim, hidden_dim, input_dim * num_input_tokens)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        #x: Dim(batch_size, num_tokens,  input_dim)
        input_tokens = self.fc1(x)
        #input_tokens: Dim(batch_size, num_tokens,  token_dim)
        output_tokens = self.encoder(input_tokens)
        #output_tokens: Dim(batch_size, num_tokens,  token_dim)
        latent_info = output_tokens[:, 0, :]
        #latent_info: Dim(batch_size, token_dim)
        mu = self.fc31(latent_info)
        log_var = self.fc32(latent_info)
        z = self.reparameterize(mu, log_var)
        #z: Dim(batch_size, latent_dim)
        recon_x = self.decoder(z)
        #recon_x: Dim(batch_size, input_dim * num_input_tokens)
        recon_x = recon_x.view(x.size())
        return recon_x, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD
    def save(self, path, optimizer=None):
        """
        Save the model state_dict and optionally the optimizer state_dict.
        """
        if optimizer:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
        else:
            torch.save(self.state_dict(), path)

    def load(self, path, optimizer=None):
        """
        Load the model state_dict and optionally the optimizer state_dict.
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
