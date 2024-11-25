from typing import Tuple, Callable
import torch.nn as nn
import numpy as np
import torch
from nav_gym.learning.modules.submodules.mlp_modules import MLP
from nav_gym.learning.modules.normalizer_module import EmpiricalNormalization

class MimicLocomotionActorCritic(nn.Module):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__()

        self.obs_shape_dict = obs_shape_dict

        self.prop_size = self.obs_shape_dict["prop"][0]
        self.exte_size = self.obs_shape_dict["exte"][0]
        self.priv_size = self.obs_shape_dict["priv"][0]
        self.mimic_obs_size =self.obs_shape_dict["mimic"][0]
        self.num_obs = self.prop_size + self.exte_size + self.priv_size + self.mimic_obs_size
        self.action_size = action_size

        self.prop_normalizer = EmpiricalNormalization(shape=[self.prop_size], until=1.0e8)
        self.exte_normalizer = EmpiricalNormalization(shape=[self.exte_size], until=1.0e8)
        self.priv_normalizer = EmpiricalNormalization(shape=[self.priv_size], until=1.0e8)
        self.mimic_normalizer = EmpiricalNormalization(shape=[self.mimic_obs_size], until=1.0e8)

        # Load configs
        self.activation_fn = get_activation(cfg["activation_fn"])

        self.scan_encoder = MLP(
            cfg["scan_encoder_shape"],
            self.activation_fn,
            self.exte_size,
            cfg["scan_latent_size"],
            init_scale=1.0 / np.sqrt(2),
        )

        if cfg["priv_encoder_shape"] is None:
            self.priv_encoder = None
            self.priv_latent_size = self.priv_size
        else:
            self.priv_encoder = MLP(
                cfg["priv_encoder_shape"],
                self.activation_fn,
                self.priv_size,
                cfg["priv_latent_size"],
                init_scale=1.0 / np.sqrt(2),
            )
            self.priv_latent_size = cfg["priv_latent_size"]

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.priv_latent_size + cfg["scan_latent_size"] + self.prop_size + self.mimic_obs_size,
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        prop, exte, priv, mimic = self.split_obs(obs)

        prop = self.prop_normalizer(prop)
        exte = self.exte_normalizer(exte)
        priv = self.priv_normalizer(priv)
        mimic = self.mimic_normalizer(mimic)


        # scan
        exte_latent = self.scan_encoder(exte)

        # priv_latent
        if self.priv_encoder is None:
            priv_latent = priv
        else:
            priv_latent = self.priv_encoder(priv)

        observation_latent = torch.cat([prop, exte_latent, priv_latent, mimic], dim=1)
        output = self.action_head(observation_latent)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.prop_size, self.exte_size, self.priv_size, self.mimic_obs_size], dim=1)
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU
    elif act_name == "selu":
        return nn.SELU
    elif act_name == "relu":
        return nn.ReLU
    elif act_name == "crelu":
        return nn.ReLU
    elif act_name == "lrelu":
        return nn.LeakyReLU
    elif act_name == "tanh":
        return nn.Tanh
    elif act_name == "sigmoid":
        return nn.Sigmoid
    else:
        print("invalid activation function!")
        return None