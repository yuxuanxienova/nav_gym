import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal


class Deterministic(nn.Module):
    def __init__(self):
        super(Deterministic, self).__init__()
        self.logits = None

    def mean(self, logits):
        # Deterministic
        return logits

    def sample(self, logits):
        self.logits = logits
        log_prob = torch.ones(logits.shape[0]).to(logits.device)
        return logits, log_prob

    def log_prob(self, samples):
        log_prob = torch.ones(samples.shape[0]).to(samples.device)
        log_prob[self.logits != samples] = 0.0
        return log_prob

    def entropy(self):
        return torch.zeros(self.logits.shape[0]).to(self.logits.device)

    def log_info(self):
        return {}


class Gaussian(nn.Module):
    def __init__(self, dim, cfg):
        super(Gaussian, self).__init__()
        self.log_std = nn.Parameter(np.log(cfg["init_std"]) * torch.ones(dim))
        self.distribution = None

    def mean(self, logits):
        return logits

    def sample(self, logits):
        # covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        # self.distribution = MultivariateNormal(logits, covariance)
        self.distribution = Normal(logits, logits * 0.0 + self.log_std.exp())

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=-1)
        return samples, log_prob

    def log_prob(self, samples):
        log_prob = self.distribution.log_prob(samples).sum(dim=-1)
        return log_prob

    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def enforce_minimum_std(self, min_std):
        current_std = self.log_std.detach().exp()
        new_log_std = torch.max(current_std, min_std.detach()).log().detach()
        self.log_std.data = new_log_std

    def enforce_maximum_std(self, max_std):
        current_std = self.log_std.detach().exp()
        new_log_std = torch.min(current_std, max_std.detach()).log().detach()
        self.log_std.data = new_log_std

    def log_info(self):
        return {"log_std": self.log_std.detach().exp().mean().item()}
