from nav_gym.learning.modules.VAE import VariationalEncoder
import torch
class HighLevelPolicy:
    def __init__(self):
        pass
    def get_command(self, obs):
        raise NotImplementedError() 
    def calcu_loss(self, obs, action, next_obs):
        raise NotImplementedError()

class HighLevelPolicy_VE(HighLevelPolicy):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        self.model = VariationalEncoder(input_dim, latent_dim, hidden_dim)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def get_command(self, input):
        #input:Dim(num_envs, input_dim)
        mu, log_var = self.model(input)
        output = self.reparameterize(mu, log_var)
        #output:Dim(num_envs, latent_dim)
        return output
    def calcu_loss(self, obs, action, next_obs):
        return self.ve.loss_function(obs, action, next_obs)