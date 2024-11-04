from learning.samplers.base import BaseSampler
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
        self.data = torch.load(load_path)
        
    def load_filtered_encoding(self, n_unfolds=5):
        # add padding to the last row of the data
        padding_data = torch.cat([self.data, self.data[:, -1, :].unsqueeze(1)], dim=1)
        unfolded_data = padding_data.unfold(1, n_unfolds, n_unfolds)
        filtered_data = unfolded_data.mean(dim=3)
        self.filtered_data = filtered_data.repeat_interleave(n_unfolds, dim=1)[:, :self.data.size(1)]
        if self.filtered_data.shape[1] != self.data.shape[1]:
            # pad the row to the length of data if the length is not the same
            self.filtered_data = torch.cat([self.filtered_data, filtered_data[:, -1, :].unsqueeze(1).repeat_interleave(self.data.shape[1] - self.filtered_data.shape[1], dim=1)], dim=1)
    
    def load_motions(self, load_path):
        self.motions = torch.load(load_path)
    
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