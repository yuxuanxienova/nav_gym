from nav_gym.learning.samplers.base import BaseSampler
import torch


class OfflineSampler(BaseSampler):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    
    def load_data(self, load_path):
        self.data = torch.load(load_path)
    
    
    def update(self, x):
        pass
        
        
    def update_curriculum(self):
        pass
    
    
    def sample(self, n_samples):
        sample_ids = torch.randint(0, self.data.size(0), (n_samples,), device=self.device, dtype=torch.long, requires_grad=False)
        return self.data[sample_ids]
    
    
class OfflineSamplerPAE(BaseSampler):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    
    def load_data(self, load_path):
        #Load From latent_params_pae.pt
        self.data = torch.load(load_path)
        #self.data: Dim: [n_motions, n_steps, n_latent_features]=[10, 169, 16]
        print("[INFO][OfflineSamplerPAE] Loaded data of shape: ", self.data.shape)
        
    def load_filtered_encoding(self, n_unfolds=5):
        #1. Add padding to the last row of the data
        #self.data: Dim: [n_motions, n_steps, n_latent_features]=[10, 169, 16]
        padding_data = torch.cat([self.data, self.data[:, -1, :].unsqueeze(1)], dim=1)
        #padding_data: Dim: [n_motions, n_steps+1, n_latent_features]=[10, 170, 16]

        #2. Unfold the data
        #(170 - 5) / 5 + 1 = 34 windows per motion.
        unfolded_data = padding_data.unfold(1, n_unfolds, n_unfolds)
        #unfolded_data: Dim: [n_motions, n_windows, window_len, n_latent_features]=[10, 34, 5, 16]

        #3. Applying Averaging to Each Window
        # compute the mean across the latent features
        filtered_data = unfolded_data.mean(dim=3)
        #filtered_data: Dim: [n_motions, n_windows, window_len]=[10, 34, 5]

        #4. Repeating the Filtered Data to Match Original Step Size
        #Transforms [10, 34, 5] to [10, 170, 5] by repeating each of the 34 windows 5 times.
        self.filtered_data = filtered_data.repeat_interleave(n_unfolds, dim=1)[:, :self.data.size(1)]
        #filtered_data: Dim: [n_motions, n_steps, window_len]=[10, 170, 5]
        if self.filtered_data.shape[1] != self.data.shape[1]:
            # pad the row to the length of data if the length is not the same
            self.filtered_data = torch.cat([self.filtered_data, filtered_data[:, -1, :].unsqueeze(1).repeat_interleave(self.data.shape[1] - self.filtered_data.shape[1], dim=1)], dim=1)
    
    def load_motions(self, load_path):
        #Load From state_transition_pae.pt
        self.motions = torch.load(load_path)
        #self.motions: Dim: [n_motions*n_trajs, n_windows, n_obs_dim, obs_horizon]=[10, 219, 21, 31]
        print("[INFO][OfflineSamplerPAE] Loaded motion of shape: ", self.motions.shape)
    
    def update(self, x):
        pass
        
        
    def update_curriculum(self):
        pass
    
    
    def sample(self, n_samples):
        motion_ids = torch.randint(0, self.data.size(0), (n_samples,), device=self.device, dtype=torch.long, requires_grad=False)
        sample_ids = torch.randint(0, self.data.size(1), (n_samples,), device=self.device, dtype=torch.long, requires_grad=False)
        return motion_ids, sample_ids # self.data[motion_ids, sample_ids], self.motions[motion_ids, sample_ids]

    def sample_curriculum(self, n_samples, curr_buf):
        motion_cand_idx = torch.arange(0, self.data.size(0), device=curr_buf.device, dtype=torch.long, requires_grad=False)
        prob = (1-curr_buf).sum(dim=0) / (curr_buf.shape[0] * self.data.size(0))
        prob = prob.clip(min=0.1, max=0.8).float()
        idx = prob.multinomial(num_samples=n_samples, replacement=True)
        motion_ids = motion_cand_idx[idx]
        sample_ids = torch.randint(0, self.data.size(1), (n_samples,), device=self.device, dtype=torch.long, requires_grad=False)
        return motion_ids, sample_ids 