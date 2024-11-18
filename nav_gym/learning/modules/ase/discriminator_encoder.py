import torch
import torch.nn as nn


class DiscriminatorEncoder(nn.Module):
    def __init__(self, history_length, num_dof, latent_dim):
        super().__init__()
        #store the parameters
        self.history_length = history_length
        self.num_dof = num_dof
        self.latent_dim = latent_dim
        #define the layers
        self.layer1 = nn.Linear(history_length*num_dof, 1024)
        self.activation1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(1024, 1024)
        self.activation2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(1024, 512)
        self.activation3 = nn.LeakyReLU()
        self.head_logits = nn.Linear(512, 1)
        self.activation_logits = nn.Sigmoid()
        self.head_mu = nn.Linear(512, latent_dim)


    
    def forward(self, amp_obs):
        #amp_obs: (num_envs, history_length, num_dof)
        #flatten the input
        x = amp_obs.view(-1, self.history_length*self.num_dof)
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        logits = self.activation_logits(self.head_logits(x))
        mu = self.head_mu(x)

        #logits: (num_envs, 1)
        #mu: (num_envs, latent_dim
        return logits, mu