import torch
import torch.nn as nn


# class DiscriminatorEncoder(nn.Module):
#     def __init__(self, history_length, num_dof, latent_dim):
#         super().__init__()
#         #store the parameters
#         self.history_length = history_length
#         self.num_dof = num_dof
#         self.latent_dim = latent_dim
#         #define the layers
#         self.layer1 = nn.Linear(history_length*num_dof, 1024)
#         self.activation1 = nn.LeakyReLU()
#         self.layer2 = nn.Linear(1024, 1024)
#         self.activation2 = nn.LeakyReLU()
#         self.layer3 = nn.Linear(1024, 512)
#         self.activation3 = nn.LeakyReLU()
#         self.head_logits = nn.Linear(512, 1)
#         self.activation_logits = nn.Sigmoid()
#         self.head_mu = nn.Linear(512, latent_dim)


    
#     def forward(self, amp_obs):
#         #amp_obs: (num_envs, history_length, num_dof)
#         #flatten the input
#         x = amp_obs.view(-1, self.history_length*self.num_dof)
#         x = self.layer1(x)
#         x = self.activation1(x)
#         x = self.layer2(x)
#         x = self.activation2(x)
#         x = self.layer3(x)
#         x = self.activation3(x)
#         logits = self.activation_logits(self.head_logits(x))
#         mu = self.head_mu(x)

#         #logits: (num_envs, 1)
#         #mu: (num_envs, latent_dim
#         return logits, mu

class DiscriminatorEncoder(nn.Module):
    def __init__(self, history_length, num_dof, latent_dim):
        super().__init__()
        # Store the parameters
        self.history_length = history_length
        self.num_dof = num_dof
        self.latent_dim = latent_dim
        
        # Define the layers
        self.layer1 = nn.Linear(history_length * num_dof, 1024)
        self.layernorm1 = nn.LayerNorm(1024)
        self.activation1 = nn.LeakyReLU()
        
        self.layer2 = nn.Linear(1024, 1024)
        self.layernorm2 = nn.LayerNorm(1024)
        self.activation2 = nn.LeakyReLU()
        
        self.layer3 = nn.Linear(1024, 512)
        self.layernorm3 = nn.LayerNorm(512)
        self.activation3 = nn.LeakyReLU()
        
        self.head_logits = nn.Linear(512, 1)
        
        self.head_mu = nn.Linear(512, latent_dim)

    def forward(self, amp_obs):
        # amp_obs: (num_envs, history_length, num_dof)
        # Flatten the input
        x = amp_obs.view(-1, self.history_length * self.num_dof)
        
        # First layer
        x = self.layer1(x)
        x = self.layernorm1(x)
        x = self.activation1(x)
        
        # Second layer
        x = self.layer2(x)
        x = self.layernorm2(x)
        x = self.activation2(x)
        
        # Third layer
        x = self.layer3(x)
        x = self.layernorm3(x)
        x = self.activation3(x)
        
        # Output layers
        logits = self.head_logits(x)
        mu = self.head_mu(x)
        
        # logits: (num_envs, 1)
        # mu: (num_envs, latent_dim)
        return logits, mu
    
    def save(self, file_path, optimizer=None):
        """
        Save the model's state dictionary and optimizer state (if provided) to the specified file path.
        
        Args:
            file_path (str): Path to the file where the model will be saved.
            optimizer (torch.optim.Optimizer, optional): Optimizer whose state will also be saved.
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'history_length': self.history_length,
            'num_dof': self.num_dof,
            'latent_dim': self.latent_dim
        }
        
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, file_path)
        print(f"[INFO] Model saved to {file_path}")

    def load(self, file_path, map_location=None, optimizer=None):
        """
        Load the model's state dictionary and optimizer state (if provided) from the specified file path.
        
        Args:
            file_path (str): Path to the file from which the model will be loaded.
            map_location (torch.device, optional): Device to map the loaded tensors.
            optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into.
        """
        checkpoint = torch.load(file_path, map_location=map_location)
        self.history_length = checkpoint['history_length']
        self.num_dof = checkpoint['num_dof']
        self.latent_dim = checkpoint['latent_dim']
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Model loaded from {file_path}")
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"[INFO] Optimizer state loaded from {file_path}")