#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
from torch.distributions import Beta

# https://github.com/hill-a/stable-baselines/issues/112
# https://keisan.casio.com/exec/system/1180573226


class BetaDistribution(nn.Module):
    def __init__(self, dim, cfg):
        super(BetaDistribution, self).__init__()
        self.output_dim = dim
        self.distribution = None
        self.alpha = None
        self.beta = None
        self.soft_plus = torch.nn.Softplus(beta=1)
        self.sigmoid = nn.Sigmoid()
        print("SCALE CHECK", cfg["scale"])

        if isinstance(cfg["scale"], tuple):
            self.scale = nn.Parameter(torch.Tensor(cfg["scale"]))
            print(self.scale)
            self.scale.requires_grad = False
        else:
            self.scale = cfg["scale"]

    def get_beta_parameters(self, logits):
        #logits: [num_envs, 2*action_dim]
        ratio = self.sigmoid(logits[:, : self.output_dim])  # (0, 1) a/(a+b) (Mean) Dim: [num_envs, action_dim]
        sum = (self.soft_plus(logits[:, self.output_dim :]) + 1) * self.scale  # (1, ~ (a+b) Dim: [num_envs, action_dim]

        alpha = ratio * sum #Dim: [num_envs, action_dim]
        beta = sum - alpha #Dim: [num_envs, action_dim]

        # For numerical stability
        alpha += 1.0e-6
        beta += 1.0e-6

        # logits_pos = self.soft_plus(logits)
        # alpha = logits_pos[:, :self.output_dim] + 1
        # beta = logits_pos[:, self.output_dim:] + 1
        return alpha, beta

    def mean(self, logits):
        return self.sigmoid(logits[:, : self.output_dim])  # Output is between 0 and 1

    def sample(self, logits):
        self.alpha, self.beta = self.get_beta_parameters(logits)
        self.distribution = Beta(self.alpha, self.beta)

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=-1)
        return samples, log_prob

    def log_prob(self, samples):
        return self.distribution.log_prob(samples).sum(dim=-1)

    def entropy(self):
        return self.distribution.entropy()

    def log_info(self):
        return {"sum": (self.alpha + self.beta).mean().item()}

# class Actor:
#     def __init__(self, architecture, distribution, obs_size, action_size, device="cpu"):
#         super(Actor, self).__init__()
#
#         self.architecture = architecture
#         self.distribution = distribution
#         self.architecture.to(device)
#         self.distribution.to(device)
#         self.device = device
#         self.obs_shape = [obs_size]
#         self.action_shape = [action_size]
#
#     def to(self, device):
#         self.architecture.to(device)
#         self.distribution.to(device)
#         self.device = device
#
#     def sample(self, obs):
#         logits = self.architecture(obs)
#         actions, log_prob = self.distribution.sample(logits)
#         return actions.detach(), log_prob.detach()
#
#     def evaluate(self, obs, actions):
#         action_mean = self.architecture(obs)
#         return self.distribution.evaluate(obs, action_mean, actions)
#
#     def parameters(self):
#         return [*self.architecture.parameters(), *self.distribution.parameters()]
#
#     def noiseless_action(self, obs):
#         logits = self.architecture(obs)
#         return self.distribution.mean(logits)
#
#     def save_deterministic_graph(self, example_input, file_name, device="cpu"):
#         # example_input = torch.randn(1, self.architecture.input_shape[0]).to(device)
#         transferred_graph = torch.jit.trace(self.architecture.to(device), example_input)
#         torch.jit.save(transferred_graph, file_name)
#         self.architecture.to(self.device)
#
#     def state_dict(self):
#         return {"architecture": self.architecture.state_dict(), "distribution": self.distribution.state_dict()}
#
#     def load_state_dict(self, state_dict):
#         self.architecture.load_state_dict(state_dict["architecture"])
#         self.distribution.load_state_dict(state_dict["distribution"])
