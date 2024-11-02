# MEMO
# https://github.com/chrischute/real-nvp/blob/df51ad570baf681e77df4d2265c0f1eb1b5b646c/models/real_nvp/real_nvp.py
# https://github.com/leggedrobotics/anomaly_navigation
# https://github.com/xqding/RealNVP/blob/master/script/RealNVP_2D.py

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class Affine_Coupling_Conditioned1D(nn.Module):
    def __init__(self, mask_reverse, in_dim, condition_dim, hidden_dim):
        super(Affine_Coupling_Conditioned1D, self).__init__()

        self.mask_reverse = mask_reverse
        self.in_dim = in_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim

        # mask to seperate positions that do not change and positions that change.
        # mask[i] = 1 means the ith position does not change.
        self.mask = checkerboard_mask1D(in_dim, self.mask_reverse)

        # layers used to compute scale in affine transformation
        self.fc1_s = nn.Linear(self.in_dim + self.condition_dim, self.hidden_dim)
        self.fc2_s = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fcout_s = nn.Linear(self.hidden_dim, self.in_dim)

        self.fc1_t = nn.Linear(self.in_dim + self.condition_dim, self.hidden_dim)
        self.fc2_t = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fcout_t = nn.Linear(self.hidden_dim, self.in_dim)

    def to(self, device):
        self.mask = self.mask.to(device)
        return self

    def _compute_s_t(self, x, c):
        # compute scaling factor using unchanged part of x with a neural network
        input = torch.cat([x, c], dim=1)
        act = torch.nn.LeakyReLU()
        out = act(self.fc1_s(input))
        out = act(self.fc2_s(out))
        s = self.fcout_s(out)

        out = act(self.fc1_t(input))
        out = act(self.fc2_t(out))
        t = self.fcout_t(out)

        # s, t = out.chunk(2, dim=1)
        return s, t

    def forward(self, x, c):
        x_fixed = x * (1 - self.mask)

        s, t = self._compute_s_t(x_fixed, c)
        s = s * self.mask
        t = t * self.mask

        exp_s = torch.exp(s)
        if torch.isnan(exp_s).any():
            print(torch.norm(x))
            print(torch.norm(c))
            raise RuntimeError("Scale factor has NaN entries")
        z = x * exp_s + t
        logdet = torch.sum(s, -1)

        return z, logdet

    def inverse(self, z, c):
        z_fixed = z * (1 - self.mask)

        s, t = self._compute_s_t(z_fixed, c)
        s = s * self.mask
        t = t * self.mask

        x = (z - t) * torch.exp(-s)
        logdet = torch.sum(-s, -1)

        return x, logdet


class Affine_Coupling(nn.Module):
    def __init__(self, mask_reverse, in_dim, hidden_dim):
        super(Affine_Coupling, self).__init__()

        self.mask_reverse = mask_reverse
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # mask to seperate positions that do not change and positions that change.
        # mask[i] = 1 means the ith position does not change.
        self.mask = checkerboard_mask1D(in_dim, self.mask_reverse)

        # layers used to compute scale in affine transformation
        self.fc1_s = nn.Linear(self.in_dim, self.hidden_dim)
        self.fc2_s = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fcout_s = nn.Linear(self.hidden_dim, self.in_dim)

        self.fc1_t = nn.Linear(self.in_dim, self.hidden_dim)
        self.fc2_t = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fcout_t = nn.Linear(self.hidden_dim, self.in_dim)
        #
        # self.scale = nn.Parameter(torch.Tensor(self.in_dim))
        # torch.nn.init.normal_(self.scale)

    def to(self, device):
        self.mask = self.mask.to(device)
        return self

    def _compute_s_t(self, x):
        # compute scaling factor using unchanged part of x with a neural network
        input = x
        act = torch.nn.LeakyReLU()
        out_act = torch.nn.Tanh()
        out = act(self.fc1_s(input))
        out = act(self.fc2_s(out))
        s = out_act(self.fcout_s(out))

        out = act(self.fc1_t(input))
        out = act(self.fc2_t(out))
        t = out_act(self.fcout_t(out))

        # s, t = out.chunk(2, dim=1)
        return s, t

    def forward(self, x):
        x_fixed = x * (1 - self.mask)

        s, t = self._compute_s_t(x_fixed)
        s = s * self.mask
        t = t * self.mask

        exp_s = torch.exp(s)
        if torch.isnan(exp_s).any():
            raise RuntimeError("Scale factor has NaN entries")
        z = x * exp_s + t
        logdet = torch.sum(s, -1)

        return z, logdet

    def inverse(self, z):
        z_fixed = z * (1 - self.mask)

        s, t = self._compute_s_t(z_fixed)
        s = s * self.mask
        t = t * self.mask

        x = (z - t) * torch.exp(-s)
        logdet = torch.sum(-s, -1)

        return x, logdet


def checkerboard_mask(height, width, reverse=False, dtype=torch.float32, device=None, requires_grad=False):
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

    if reverse:
        mask = 1 - mask

    # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.view(1, 1, height, width)
    return mask


def checkerboard_mask1D(width, reverse=False, dtype=torch.float32, requires_grad=False):
    checkerboard = [j % 2 for j in range(width)]
    mask = torch.tensor(checkerboard, dtype=dtype, requires_grad=requires_grad)

    if reverse:
        mask = 1 - mask

    mask = mask.view(1, width)
    return mask


class ConditionalRealNVP(nn.Module):
    def __init__(self, in_dim, condition_dim, hidden_dim):
        super(ConditionalRealNVP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(self.in_dim), torch.eye(self.in_dim))

        self.affine_couplings = nn.ModuleList(
            [
                Affine_Coupling_Conditioned1D(True, in_dim, condition_dim, hidden_dim),
                Affine_Coupling_Conditioned1D(False, in_dim, condition_dim, hidden_dim),
                Affine_Coupling_Conditioned1D(True, in_dim, condition_dim, hidden_dim),
                Affine_Coupling_Conditioned1D(False, in_dim, condition_dim, hidden_dim),
            ]
        )

    def to(self, device):
        super().to(device)
        self.affine_couplings = nn.ModuleList([(aff_map.to(device)) for aff_map in self.affine_couplings])
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(self.in_dim, device=device), torch.eye(self.in_dim, device=device)
        )
        return self

    def forward(self, y, x_c):
        x = y
        condition = torch.reshape(x_c, [x_c.shape[0], -1])

        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            x, logdet = self.affine_couplings[i](x, condition)
            logdet_tot = logdet_tot + logdet

        # ## a normalization layer is added such that the observed variables is within
        # ## the range of [-4, 4].
        # logdet = torch.sum(torch.log(torch.abs(4*(1-(torch.tanh(x))**2))), -1)
        # x = 4*torch.tanh(x)
        # logdet_tot = logdet_tot + logdet

        return x, logdet_tot

    def inverse(self, x, x_c):
        y = x
        logdet_tot = 0
        condition = torch.reshape(x_c, [x_c.shape[0], -1])

        # inverse the normalization layer
        # logdet = torch.sum(torch.log(torch.abs(1.0/4.0* 1/(1-(y/4)**2))), -1)
        # y = 0.5*torch.log((1+y/4)/(1-y/4))
        # logdet_tot = logdet_tot + logdet

        # inverse affine coupling layers
        for i in range(len(self.affine_couplings) - 1, -1, -1):
            y, logdet = self.affine_couplings[i].inverse(y, condition)
            logdet_tot = logdet_tot + logdet

        return y, logdet_tot

    def log_prob(self, y, x_c):  # x: latent
        x, logp = self.forward(y, x_c)
        return self.prior.log_prob(x) + logp


class RealNVP(nn.Module):
    def __init__(self, in_dim, hidden_dim, min_val=None, max_val=None, tanh_output=False):
        super(RealNVP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(self.in_dim), torch.eye(self.in_dim))

        self.tanh_output = tanh_output

        # Z -- forward --> Y
        # SCALE Y

        if max_val is None:
            self.min_val = nn.Parameter(torch.zeros(self.in_dim), requires_grad=False)
            self.max_val = nn.Parameter(torch.ones(self.in_dim), requires_grad=False)
        else:
            self.max_val = nn.Parameter(torch.tensor(max_val, dtype=torch.float32), requires_grad=False)
            self.min_val = nn.Parameter(torch.tensor(min_val, dtype=torch.float32), requires_grad=False)

        self.scale = nn.Parameter((self.max_val - self.min_val) * 0.5, requires_grad=False)

        self.affine_couplings = nn.ModuleList(
            [
                Affine_Coupling(True, in_dim, hidden_dim),
                Affine_Coupling(False, in_dim, hidden_dim),
                # Affine_Coupling(True, in_dim, hidden_dim),
                # Affine_Coupling(False, in_dim, hidden_dim),
                Affine_Coupling(True, in_dim, hidden_dim),
                Affine_Coupling(False, in_dim, hidden_dim),
            ]
        )

    def to(self, device):
        super().to(device)
        self.affine_couplings = nn.ModuleList([(aff_map.to(device)) for aff_map in self.affine_couplings])
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(self.in_dim, device=device), torch.eye(self.in_dim, device=device)
        )

        return self

    def forward(self, z):
        y = z
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet

        if self.tanh_output:
            log_one_minus_tanhy_square = np.log(2.0) - y - torch.nn.functional.softplus(-2.0 * y)
            y = torch.tanh(y)

            # logdet_tot = logdet_tot + torch.sum(torch.log((1.0-y.pow(2)) + 1e-8), dim=1)
            logdet_tot = logdet_tot + torch.sum(log_one_minus_tanhy_square, dim=1)
            # torch.sum(torch.log((1 - y.pow(2)) + 1e-6), dim=1)

        # scale
        y = y + 1.0
        y = y * self.scale + self.min_val

        return y, logdet_tot

    def inverse(self, y):
        z = y
        logdet_tot = 0

        # unscale
        z = (z - self.min_val) / self.scale - 1.0  # in [-1, 1]
        # # inverse the normalization layer
        # logdet = torch.sum(torch.log(torch.abs(1.0/4.0* 1/(1-(y/4)**2))), -1)
        # y = 0.5*torch.log((1+y/4)/(1-y/4))
        # logdet_tot = logdet_tot + logdet

        if self.tanh_output:
            clip_val = 1.0 - 1e-6  # to avoid 0 denominator
            z = torch.clip(z, -clip_val, clip_val)
            # z = 0.5 * (z.log1p() - (-z).log1p())  # numerically stable impl
            # logdet_tot = logdet_tot + torch.sum(torch.log(1.0 / (1.0- z.pow(2) + 1e-8)), dim=1)

            z = 0.5 * torch.log((1 + z) / (1 - z))
            log_one_minus_tanhz_square = np.log(2.0) - z - torch.nn.functional.softplus(-2.0 * z)
            logdet_tot = logdet_tot - torch.sum(log_one_minus_tanhz_square, dim=1)

        # if torch.sum(torch.isnan(z)) > 0:
        #     print("WTF")
        #     exit()
        #
        # if torch.sum(torch.isnan(logdet_tot)) > 0:
        #     print( torch.sum(torch.log(1.0 / (1.0- z.pow(2) + 1e-8)), dim=1))
        #     print(1.0- z.pow(2))
        #     print("WTF?")
        #     exit()

        # inverse affine coupling layers
        for i in range(len(self.affine_couplings) - 1, -1, -1):
            z, logdet = self.affine_couplings[i].inverse(z)
            logdet_tot = logdet_tot + logdet

        return z, logdet_tot

    def log_prob(self, y):  # z: latent
        z, logp = self.inverse(y)

        return self.prior.log_prob(z) + logp


class FlowActorDistribution(nn.Module):
    def __init__(self, dim, nvp_hidden_dim=16, init_std=1.0, min_val=None, max_val=None, tanh_output=False):
        super(FlowActorDistribution, self).__init__()
        self.log_std = nn.Parameter(np.log(init_std) * torch.ones(dim))
        self.dim = dim
        self.flow = RealNVP(self.dim, nvp_hidden_dim, min_val, max_val, tanh_output)

        self.prior = None

    def freezeNVP(self):
        self.flow.requires_grad_(False)

    def to(self, device):
        super().to(device)
        self.flow.to(device)
        return self

    def forward(self, logits):
        nvp_generated, logdet = self.flow.forward(logits)

        return nvp_generated

    def sample(self, logits):
        nvp_covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        self.prior = MultivariateNormal(logits, nvp_covariance)
        nvp_x = self.prior.sample()
        nvp_x_logp = self.prior.log_prob(nvp_x)
        nvp_generated, logdet = self.flow.forward(nvp_x)

        nvp_logp = nvp_x_logp - logdet
        return nvp_generated, nvp_logp

    def rsample(self, logits):
        nvp_covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        self.prior = MultivariateNormal(logits, nvp_covariance)
        nvp_x = self.prior.rsample()
        nvp_x_logp = self.prior.log_prob(nvp_x)
        nvp_generated, logdet = self.flow.forward(nvp_x)

        nvp_logp = nvp_x_logp - logdet
        return nvp_generated, nvp_logp

    def evaluate(self, inputs, logits, outputs):
        nvp_covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        self.prior = MultivariateNormal(logits, nvp_covariance)

        nvp_x, logdet = self.flow.inverse(outputs)
        nvp_logp = self.prior.log_prob(nvp_x) + logdet

        # entropy =  self.prior.entropy()

        # Approx entropy using logp
        # https://www.reddit.com/r/reinforcementlearning/comments/n5beqy/question_entropy_of_transformed_tanh_distribution/
        entropy = -nvp_logp

        # return log_prob, entropy
        return nvp_logp, entropy

    def entropy(self):
        return self.prior.entropy()  # poor approximation

    def enforce_minimum_std(self, min_std):
        current_std = self.log_std.detach().exp()
        new_log_std = torch.max(current_std, min_std.detach()).log().detach()
        self.log_std.data = new_log_std

    def enforce_maximum_std(self, max_std):
        current_std = self.log_std.detach().exp()
        new_log_std = torch.min(current_std, max_std.detach()).log().detach()
        self.log_std.data = new_log_std
