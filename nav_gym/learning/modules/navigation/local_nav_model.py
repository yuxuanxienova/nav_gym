from typing import Tuple, Callable
import torch.nn as nn
import numpy as np
import torch
from nav_gym.learning.modules.submodules.cnn_modules import CnnSPP, SimpleCNN, Pool1DConv, SimpleCNN2
from nav_gym.learning.modules.submodules.mlp_modules import MLP
from nav_gym.learning.modules.navigation.graph_models import SimplePointNet, MultiHeadAttention, DGCNN, Conv3DNet
from nav_gym.learning.modules.normalizer_module import EmpiricalNormalization


# class SimpleNavPolicy(nn.Module):
#     def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
#         super().__init__()

#         self.cfg = cfg
#         self.obs_shape_dict = obs_shape_dict

#         # self.command_size = self.obs_shape_dict["command"][0]
#         self.h_map_shape = self.obs_shape_dict["ext"]
#         self.h_map_size = self.h_map_shape.numel()
#         self.prop_size = self.obs_shape_dict["prop"][0]
#         self.history_shape = self.obs_shape_dict["history"]
#         self.history_obs_size = self.history_shape.numel()

#         self.activation_fn = get_activation(cfg["activation_fn"])

#         self.num_obs = self.h_map_size + self.prop_size + self.history_obs_size
#         self.action_size = action_size

#         # self.num_obs_normalizer = self.h_map_size + self.prop_size

#         # self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)
#         self.obs_normalizer_prop = EmpiricalNormalization(shape=[self.prop_size], until=1.0e8)
#         self.obs_normalizer_scan = EmpiricalNormalization(shape=[self.h_map_size], until=1.0e8)

#         grid_map_length = int(np.sqrt(self.h_map_shape[-1]))
#         self.scan_encoder = SimpleCNN2([self.h_map_shape[0], grid_map_length, grid_map_length],
#                                         cfg["scan_cnn_channels"],
#                                         cfg["scan_cnn_fc_shape"],
#                                         cfg["scan_latent_size"],
#                                         self.activation_fn)


#         self.history_encoder = Pool1DConv(self.history_shape,
#                                           cfg["history_channels"],
#                                           cfg["history_fc_shape"],
#                                           cfg["history_latent_size"],
#                                           cfg["history_kernel_size"],
#                                           activation_fn=self.activation_fn)

#         self.action_head = MLP(
#             cfg["output_mlp_size"],
#             self.activation_fn,
#             self.prop_size+ cfg["scan_latent_size"] + cfg["history_latent_size"],
#             self.action_size,
#             init_scale=1.0 / np.sqrt(2),
#         )



#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         prop, scan,  history = self.split_obs(obs)
#         # obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
#         # obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)

#         prop = self.obs_normalizer_prop(prop)
#         scan = self.obs_normalizer_scan(scan)

#         # scan
#         scan_latent = self.scan_encoder(scan)

#         # history
#         history_latent = self.history_encoder(history)

#         concat_feature = torch.cat([scan_latent, prop, history_latent], dim=1)
#         output = self.action_head(concat_feature)
#         return output

#     def split_obs(self, obs):
#         return torch.split(obs, [ self.prop_size, self.h_map_size, self.history_obs_size], dim=1)


class NavPolicyWithMemory(nn.Module):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.obs_shape_dict = obs_shape_dict

        #1. get the shape of the observation
        self.prop_size = self.obs_shape_dict["prop"][0]

        self.h_map_shape = self.obs_shape_dict["ext"]
        self.h_map_size = self.h_map_shape.numel()
        self.grid_map_length = int(np.sqrt(self.h_map_shape[-1]))

        self.history_shape = self.obs_shape_dict["history"]
        self.history_obs_size = self.history_shape.numel()

        
        self.memory_shape = self.obs_shape_dict["memory"]
        self.memory_size = self.memory_shape.numel()
        self.num_nodes = self.memory_shape[0]

        self.num_obs = self.h_map_size + self.prop_size + self.history_obs_size + self.memory_size
        self.action_size = action_size
        #2, Initialize the modules
        self.activation_fn = get_activation(cfg["activation_fn"])

        self.obs_normalizer_prop = EmpiricalNormalization(shape=[self.prop_size], until=1.0e8)
        self.obs_normalizer_scan = EmpiricalNormalization(shape=[self.h_map_size], until=1.0e8)

        


        self.ext_encoder = SimpleCNN2([1, self.grid_map_length, self.grid_map_length],
                                        cfg["scan_cnn_channels"],
                                        cfg["scan_cnn_fc_shape"],
                                        cfg["scan_latent_size"],
                                        self.activation_fn)


        self.history_encoder = Pool1DConv(self.history_shape,
                                          cfg["history_channels"],
                                          cfg["history_fc_shape"],
                                          cfg["history_latent_size"],
                                          cfg["history_kernel_size"],
                                          activation_fn=self.activation_fn)
        self.memory_encoder = SimplePointNet([self.num_nodes, self.memory_shape[1]],
                                         cfg["pointnet_channels"],
                                         cfg["pointnet_fc_shape"],
                                         cfg["aggregator_latent_size"],
                                         activation_fn=self.activation_fn)

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.prop_size+ cfg["scan_latent_size"] + cfg["history_latent_size"] + cfg["aggregator_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        self.num_envs = obs.shape[0]
        #obs: [batch, obs_size]
        #obs_size = prop_size + h_map_size + history_obs_size + memory_size

        #1. split the observation
        prop, ext,  history, memory = self.split_obs(obs)
        #2. normalize the observation
        prop = self.obs_normalizer_prop(prop)
        ext = self.obs_normalizer_scan(ext)
        #3. encode the observation
        # ext
        ext_latent = self.ext_encoder(ext)
        # history
        history_latent = self.history_encoder(history)
        # memory
        memory_feature = memory.reshape(self.num_envs, self.num_nodes, -1)
        pointnet_feature = self.memory_encoder(memory_feature)

        concat_feature = torch.cat([ext_latent, prop, history_latent, pointnet_feature], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [ self.prop_size, self.h_map_size, self.history_obs_size, self.memory_size], dim=1)


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
    elif act_name == "softsign":
        return nn.Softsign
    else:
        print("invalid activation function!")
        return None
