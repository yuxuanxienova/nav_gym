from typing import Tuple, Callable
import torch.nn as nn
import numpy as np
import torch
from nav_gym.learning.modules.submodules.cnn_modules import CnnSPP, SimpleCNN, Pool1DConv, SimpleCNN2
from nav_gym.learning.modules.submodules.mlp_modules import MLP
from nav_gym.learning.modules.navigation.graph_models import SimplePointNet, MultiHeadAttention, DGCNN, Conv3DNet
from nav_gym.learning.modules.normalizer_module import EmpiricalNormalization


class NavPolicyBase(nn.Module):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.obs_shape_dict = obs_shape_dict

        self.h_map_size = self.obs_shape_dict["ext"][0]
        self.node_p_shape = self.obs_shape_dict["node_p"]
        self.node_m_shape = self.obs_shape_dict["node_m"]
        self.history_shape = self.obs_shape_dict["history"]

        self.node_p_size = self.node_p_shape.numel()
        self.node_m_size = self.node_m_shape.numel()
        self.history_obs_size = self.history_shape.numel()

        self.num_nodes = self.node_p_shape[0]

        self.num_obs = self.h_map_size + self.node_p_size + self.node_m_size + self.history_obs_size
        self.num_obs_normalizer = self.h_map_size
        self.action_size = action_size

        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)

        # Load configs
        self.activation_fn = get_activation(cfg["activation_fn"])

        # self.scan_encoder = MLP(
        #     cfg["scan_encoder_shape"],
        #     self.activation_fn,
        #     self.h_map_size,
        #     cfg["scan_latent_size"],
        #     init_scale=1.0 / np.sqrt(2),
        # )

        grid_map_length = int(np.sqrt(self.h_map_size))
        self.scan_encoder = SimpleCNN2([1, grid_map_length, grid_map_length],
                                        cfg["scan_cnn_channels"],
                                        cfg["scan_cnn_fc_shape"],
                                        cfg["scan_latent_size"],
                                        self.activation_fn)

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            cfg["scan_latent_size"] + cfg["aggregator_latent_size"] + cfg["history_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )

        self.history_encoder = Pool1DConv(self.history_shape,
                                          cfg["history_channels"],
                                          cfg["history_fc_shape"],
                                          cfg["history_latent_size"],
                                          cfg["history_conv_kernel_size"],
                                          activation_fn=self.activation_fn)

        # self.graph_map_encoder = SimpleCNN(self.node_m_shape[1:],
        #                                    cfg["graph_map_channels"],
        #                                    cfg["graph_map_fc_shape"],
        #                                    cfg["graph_map_latent_size"],
        #                                    activation_fn=self.activation_fn) # TODO: later maybe

        self.graph_map_encoder = MLP(cfg["graph_map_fc_shape"],
                                     self.activation_fn,
                                     self.node_m_shape[1],
                                     cfg["graph_map_latent_size"],
                                     init_scale=1.0 / np.sqrt(2),
                                     )

        if cfg["aggregator_type"] == "PN":
            self.aggregator = SimplePointNet([self.num_nodes, cfg["graph_map_latent_size"] + self.node_p_shape[1]],
                                            cfg["pointnet_channels"],
                                            cfg["pointnet_fc_shape"],
                                            cfg["aggregator_latent_size"],
                                            activation_fn=self.activation_fn)
        elif cfg["aggregator_type"] == "Attention":
            self.aggregator = MultiHeadAttention([self.num_nodes, cfg["graph_map_latent_size"] + self.node_p_shape[1]],
                                            cfg["aggregator_encoder_dim"],
                                            cfg["aggregator_encoder_dim"],
                                            cfg["aggregaor_num_heads"])
        elif cfg["aggregator_type"] == "DGCNN":
            self.aggregator = DGCNN(cfg["graph_map_latent_size"] + self.node_p_shape[1],
                                    cfg["aggregator_embedding_dim"],
                                    cfg["aggregator_k"],
                                    cfg["aggregaor_output"])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
        obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)
        scan, node_p, node_m, history = self.split_obs(obs)

        # scan
        scan_latent = self.scan_encoder(scan)

        # history
        history_latent = self.history_encoder(history)

        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_m_feature = self.graph_map_encoder(node_m)

        # aggregation
        node_m_feature = node_m_feature.reshape(scan_latent.shape[0], self.num_nodes, -1)
        node_p_feature = node_p.reshape(scan_latent.shape[0], self.num_nodes, -1)
        concat_node_feature = torch.cat([node_m_feature, node_p_feature], dim=2)

        pointnet_feature = self.aggregator(concat_node_feature)

        concat_feature = torch.cat([scan_latent, history_latent, pointnet_feature], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.h_map_size, self.node_p_size, self.node_m_size, self.history_obs_size], dim=1)


class NavPolicyVelodyne(NavPolicyBase):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__(obs_shape_dict, action_size, cfg, **kwargs)
        self.velodyne_scan_shape = self.obs_shape_dict["velodyne_scan"]
        self.velodyne_scan_obs_size = self.velodyne_scan_shape.numel()
        self.num_obs = self.h_map_size + self.velodyne_scan_obs_size + self.node_p_size + self.node_m_size + self.history_obs_size
        self.num_obs_normalizer = self.h_map_size + self.velodyne_scan_obs_size
        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)

        # self.velodyne_scan_encoder = Pool1DConv(self.velodyne_scan_shape,
        #                                         cfg["velodyne_scan_channels"],
        #                                         cfg["velodyne_scan_fc_shape"],
        #                                         cfg["velodyne_scan_latent_size"],
        #                                         cfg["velodyne_scan_conv_kernel_size"],
        #                                         activation_fn=self.activation_fn)
        self.velodyne_scan_encoder = MLP(
            cfg["velodyne_scan_fc_shape"],
            self.activation_fn,
            self.velodyne_scan_obs_size,
            cfg["velodyne_scan_latent_size"],
            init_scale=1.0 / np.sqrt(2),
        )

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            cfg["scan_latent_size"] + cfg["velodyne_scan_latent_size"] + cfg["aggregator_latent_size"] + cfg[
                "history_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
        obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)
        scan, velodyne_scan, node_p, node_m, history = self.split_obs(obs)

        # scan
        scan_latent = self.scan_encoder(scan)

        # scan
        velodyne_scan_latent = self.velodyne_scan_encoder(velodyne_scan)

        # history
        history_latent = self.history_encoder(history)

        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_m_feature = self.graph_map_encoder(node_m)

        # aggregation
        node_m_feature = node_m_feature.reshape(scan_latent.shape[0], self.num_nodes, -1)
        node_p_feature = node_p.reshape(scan_latent.shape[0], self.num_nodes, -1)
        concat_node_feature = torch.cat([node_m_feature, node_p_feature], dim=2)

        pointnet_feature = self.aggregator(concat_node_feature)

        concat_feature = torch.cat([scan_latent, velodyne_scan_latent, history_latent, pointnet_feature], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.h_map_size, self.velodyne_scan_obs_size, self.node_p_size, self.node_m_size,
                                 self.history_obs_size], dim=1)


class NavPolicyVelodynePt(NavPolicyBase):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__(obs_shape_dict, action_size, cfg, **kwargs)

        self.command_size = self.obs_shape_dict["command"][0]

        self.velodyne_scan_shape = self.obs_shape_dict["velodyne_scan"]
        self.velodyne_scan_obs_size = self.velodyne_scan_shape.numel()
        self.num_obs = self.command_size + self.h_map_size + self.velodyne_scan_obs_size + self.node_p_size + self.node_m_size + self.history_obs_size
        self.num_obs_normalizer = self.command_size + self.h_map_size + self.velodyne_scan_obs_size
        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)

        # self.velodyne_scan_encoder = Pool1DConv(self.velodyne_scan_shape,
        #                                         cfg["velodyne_scan_channels"],
        #                                         cfg["velodyne_scan_fc_shape"],
        #                                         cfg["velodyne_scan_latent_size"],
        #                                         cfg["velodyne_scan_conv_kernel_size"],
        #                                         activation_fn=self.activation_fn)
        self.velodyne_scan_encoder = MLP(
            cfg["velodyne_scan_fc_shape"],
            self.activation_fn,
            self.velodyne_scan_obs_size,
            cfg["velodyne_scan_latent_size"],
            init_scale=1.0 / np.sqrt(2),
        )

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.command_size
            + cfg["scan_latent_size"]
            + cfg["velodyne_scan_latent_size"]
            + cfg["aggregator_latent_size"]
            + cfg["history_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
        obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)
        commands, scan, velodyne_scan, node_p, node_m, history = self.split_obs(obs)

        # scan
        scan_latent = self.scan_encoder(scan)

        # scan
        velodyne_scan_latent = self.velodyne_scan_encoder(velodyne_scan)

        # history
        history_latent = self.history_encoder(history)

        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_m_feature = self.graph_map_encoder(node_m)

        # aggregation
        node_m_feature = node_m_feature.reshape(scan_latent.shape[0], self.num_nodes, -1)
        node_p_feature = node_p.reshape(scan_latent.shape[0], self.num_nodes, -1)
        concat_node_feature = torch.cat([node_m_feature, node_p_feature], dim=2)

        pointnet_feature = self.aggregator(concat_node_feature)

        concat_feature = torch.cat([commands, scan_latent, velodyne_scan_latent, history_latent, pointnet_feature], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.command_size, self.h_map_size, self.velodyne_scan_obs_size, self.node_p_size, self.node_m_size,
                                 self.history_obs_size], dim=1)



class NavPolicyPt(NavPolicyBase):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__(obs_shape_dict, action_size, cfg)

        self.command_size = self.obs_shape_dict["command"][0]
        self.num_obs = self.command_size + self.h_map_size + self.node_p_size + self.node_m_size + self.history_obs_size
        self.num_obs_normalizer = self.command_size + self.h_map_size
        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)
        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.command_size + cfg["scan_latent_size"] + cfg["aggregator_latent_size"] + cfg["history_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )


        # EXPERIMENTAL
        self.graph_feature_encoder = MLP(cfg["graph_feature_fc_shape"],
                                         self.activation_fn,
                                         cfg["graph_map_latent_size"] + self.node_p_shape[1] + self.command_size,
                                         cfg["graph_feature_latent_size"],
                                         init_scale=1.0 / np.sqrt(2),
                                         )

        self.aggregator = SimplePointNet([self.num_nodes, cfg["graph_feature_latent_size"]],
                                         cfg["pointnet_channels"],
                                         cfg["pointnet_fc_shape"],
                                         cfg["aggregator_latent_size"],
                                         activation_fn=self.activation_fn)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
        obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)
        commands, scan, node_p, node_m, history = self.split_obs(obs)

        # scan
        scan_latent = self.scan_encoder(scan)

        # history
        history_latent = self.history_encoder(history)

        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_m_feature = self.graph_map_encoder(node_m)

        # aggregation
        node_m_feature = node_m_feature.reshape(scan_latent.shape[0], self.num_nodes, -1)
        # node_p_feature = node_p.reshape(scan_latent.shape[0], self.num_nodes, -1)

        command_repeated = commands.repeat(1, self.num_nodes).reshape(-1, self.num_nodes, commands.shape[1])
        node_p = node_p.reshape(node_p.shape[0], self.num_nodes, -1)
        node_p_feature = torch.cat([node_p, command_repeated], dim=2)  # shape = (batch_size, num_nodes, node_p_size + command_size)

        concat_node_features = torch.cat([node_m_feature, node_p_feature], dim=2)


        node_feature = self.graph_feature_encoder(concat_node_features.reshape(obs.shape[0] * self.num_nodes, -1))
        node_feature = node_feature.reshape(obs.shape[0], self.num_nodes, -1)

        pointnet_feature = self.aggregator(node_feature)

        concat_feature = torch.cat([commands, scan_latent, history_latent, pointnet_feature], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.command_size, self.h_map_size, self.node_p_size, self.node_m_size,
                                 self.history_obs_size], dim=1)



class NavPolicyPt2(NavPolicyBase):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__(obs_shape_dict, action_size, cfg)

        self.command_size = self.obs_shape_dict["command"][0]
        self.num_obs_normalizer = self.command_size + self.h_map_size
        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)


        self.near_nodes_shape = self.obs_shape_dict["near_nodes"]
        self.near_nodes_size = self.near_nodes_shape.numel()

        self.num_obs = self.command_size + self.h_map_size + self.node_p_size + self.node_m_size + self.history_obs_size
        self.num_obs += self.near_nodes_size


        self.near_nodes_encoder = SimplePointNet([self.near_nodes_shape[0], self.near_nodes_shape[1]],
                                                    cfg["near_nodes_channels"],
                                                    cfg["near_nodes_fc_shape"],
                                                    cfg["near_nodes_latent_size"],
                                                    activation_fn=self.activation_fn)


        # EXPERIMENTAL
        self.graph_feature_encoder = MLP(cfg["graph_feature_fc_shape"],
                                         self.activation_fn,
                                         cfg["graph_map_latent_size"] + self.node_p_shape[1] + self.command_size,
                                         cfg["graph_feature_latent_size"],
                                         init_scale=1.0 / np.sqrt(2),
                                         )

        self.aggregator = SimplePointNet([self.num_nodes, cfg["graph_feature_latent_size"]],
                                         cfg["pointnet_channels"],
                                         cfg["pointnet_fc_shape"],
                                         cfg["aggregator_latent_size"],
                                         activation_fn=self.activation_fn)


        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.command_size
            + cfg["scan_latent_size"]
            + cfg["aggregator_latent_size"]
            + cfg["history_latent_size"]
            + cfg["near_nodes_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )



    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
        obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)
        commands, scan, node_p, node_m, history, near_nodes = self.split_obs(obs)

        # scan
        scan_latent = self.scan_encoder(scan)

        # history
        history_latent = self.history_encoder(history)


        # nearby nodes
        near_nodes_latent = self.near_nodes_encoder(near_nodes)

        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_m_feature = self.graph_map_encoder(node_m)

        # aggregation
        node_m_feature = node_m_feature.reshape(scan_latent.shape[0], self.num_nodes, -1)
        # node_p_feature = node_p.reshape(scan_latent.shape[0], self.num_nodes, -1)

        command_repeated = commands.repeat(1, self.num_nodes).reshape(-1, self.num_nodes, commands.shape[1])
        node_p = node_p.reshape(node_p.shape[0], self.num_nodes, -1)
        node_p_feature = torch.cat([node_p, command_repeated], dim=2)  # shape = (batch_size, num_nodes, node_p_size + command_size)

        concat_node_features = torch.cat([node_m_feature, node_p_feature], dim=2)


        node_feature = self.graph_feature_encoder(concat_node_features.reshape(obs.shape[0] * self.num_nodes, -1))
        node_feature = node_feature.reshape(obs.shape[0], self.num_nodes, -1)

        pointnet_feature = self.aggregator(node_feature)

        concat_feature = torch.cat([commands, scan_latent, history_latent, pointnet_feature, near_nodes_latent], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.command_size, self.h_map_size, self.node_p_size, self.node_m_size,
                                 self.history_obs_size, self.near_nodes_size], dim=1)


class NavPolicyPt3(NavPolicyBase):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__(obs_shape_dict, action_size, cfg)

        self.command_size = self.obs_shape_dict["command"][0]
        self.num_obs_normalizer = self.command_size + self.h_map_size
        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)

        self.num_obs = self.command_size + self.h_map_size + self.node_p_size + self.node_m_size + self.history_obs_size


        # EXPERIMENTAL
        self.graph_feature_encoder = MLP(cfg["graph_feature_fc_shape"],
                                         self.activation_fn,
                                         cfg["graph_map_latent_size"] + self.node_p_shape[1] + self.command_size,
                                         cfg["graph_feature_latent_size"],
                                         init_scale=1.0 / np.sqrt(2),
                                         )

        self.aggregator = SimplePointNet([self.num_nodes, cfg["graph_feature_latent_size"]],
                                         cfg["pointnet_channels"],
                                         cfg["pointnet_fc_shape"],
                                         cfg["aggregator_latent_size"],
                                         activation_fn=self.activation_fn)


        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.command_size
            + cfg["scan_latent_size"]
            + cfg["aggregator_latent_size"]
            + cfg["history_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )



    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
        obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)
        commands, scan, node_p, node_m, history = self.split_obs(obs)

        # scan
        scan_latent = self.scan_encoder(scan)

        # history
        history_latent = self.history_encoder(history)



        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_m_feature = self.graph_map_encoder(node_m)

        # aggregation
        node_m_feature = node_m_feature.reshape(scan_latent.shape[0], self.num_nodes, -1)
        # node_p_feature = node_p.reshape(scan_latent.shape[0], self.num_nodes, -1)

        command_repeated = commands.repeat(1, self.num_nodes).reshape(-1, self.num_nodes, commands.shape[1])
        node_p = node_p.reshape(node_p.shape[0], self.num_nodes, -1)
        node_p_feature = torch.cat([node_p, command_repeated], dim=2)  # shape = (batch_size, num_nodes, node_p_size + command_size)

        concat_node_features = torch.cat([node_m_feature, node_p_feature], dim=2)


        node_feature = self.graph_feature_encoder(concat_node_features.reshape(obs.shape[0] * self.num_nodes, -1))
        node_feature = node_feature.reshape(obs.shape[0], self.num_nodes, -1)

        pointnet_feature = self.aggregator(node_feature)

        concat_feature = torch.cat([commands, scan_latent, history_latent, pointnet_feature], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.command_size, self.h_map_size, self.node_p_size, self.node_m_size,
                                 self.history_obs_size], dim=1)


class NavPolicyPt2NoFeatures(NavPolicyBase):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        obs_shape_dict["node_m"] = obs_shape_dict["node_p"]
        super().__init__(obs_shape_dict, action_size, cfg)

        self.command_size = self.obs_shape_dict["command"][0]
        self.num_obs_normalizer = self.command_size + self.h_map_size
        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)


        self.near_nodes_shape = self.obs_shape_dict["near_nodes"]
        self.near_nodes_size = self.near_nodes_shape.numel()

        self.num_obs = self.command_size + self.h_map_size + self.node_p_size + self.history_obs_size
        self.num_obs += self.near_nodes_size


        self.near_nodes_encoder = SimplePointNet([self.near_nodes_shape[0], self.near_nodes_shape[1]],
                                                    cfg["near_nodes_channels"],
                                                    cfg["near_nodes_fc_shape"],
                                                    cfg["near_nodes_latent_size"],
                                                    activation_fn=self.activation_fn)


        # EXPERIMENTAL
        self.graph_map_encoder = None
        self.graph_feature_encoder = MLP(cfg["graph_feature_fc_shape"],
                                         self.activation_fn,
                                         self.node_p_shape[1] + self.command_size,
                                         cfg["graph_feature_latent_size"],
                                         init_scale=1.0 / np.sqrt(2),
                                         )

        self.aggregator = SimplePointNet([self.num_nodes, cfg["graph_feature_latent_size"]],
                                         cfg["pointnet_channels"],
                                         cfg["pointnet_fc_shape"],
                                         cfg["aggregator_latent_size"],
                                         activation_fn=self.activation_fn)


        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.command_size
            + cfg["scan_latent_size"]
            + cfg["aggregator_latent_size"]
            + cfg["history_latent_size"]
            + cfg["near_nodes_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )



    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
        obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)
        commands, scan, node_p, history, near_nodes = self.split_obs(obs)

        # scan
        scan_latent = self.scan_encoder(scan)

        # history
        history_latent = self.history_encoder(history)


        # nearby nodes
        near_nodes_latent = self.near_nodes_encoder(near_nodes)
        # node_p_feature = node_p.reshape(scan_latent.shape[0], self.num_nodes, -1)

        command_repeated = commands.repeat(1, self.num_nodes).reshape(-1, self.num_nodes, commands.shape[1])
        node_p = node_p.reshape(node_p.shape[0], self.num_nodes, -1)
        node_p_feature = torch.cat([node_p, command_repeated], dim=2)  # shape = (batch_size, num_nodes, node_p_size + command_size)

        # concat_node_features = torch.cat([node_m_feature, node_p_feature], dim=2)


        node_feature = self.graph_feature_encoder(node_p_feature.reshape(obs.shape[0] * self.num_nodes, -1))
        node_feature = node_feature.reshape(obs.shape[0], self.num_nodes, -1)

        pointnet_feature = self.aggregator(node_feature)

        concat_feature = torch.cat([commands, scan_latent, history_latent, pointnet_feature, near_nodes_latent], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.command_size, self.h_map_size, self.node_p_size,
                                 self.history_obs_size, self.near_nodes_size], dim=1)


class NavPolicyOnlyRays(nn.Module):

    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.obs_shape_dict = obs_shape_dict

        self.node_p_shape = self.obs_shape_dict["node_p"]
        self.node_m_shape = self.obs_shape_dict["node_m"]
        self.history_shape = self.obs_shape_dict["history"]

        self.node_p_size = self.node_p_shape.numel()
        self.node_m_size = self.node_m_shape.numel()
        self.history_obs_size = self.history_shape.numel()

        self.num_nodes = self.node_p_shape[0]

        self.velodyne_scan_shape = self.obs_shape_dict["velodyne_scan"]
        self.velodyne_scan_obs_size = self.velodyne_scan_shape.numel()
        self.num_obs = self.velodyne_scan_obs_size + self.node_p_size + self.node_m_size + self.history_obs_size
        self.num_obs_normalizer = self.velodyne_scan_obs_size

        self.action_size = action_size

        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)

        # Load configs
        self.activation_fn = get_activation(cfg["activation_fn"])


        self.history_encoder = Pool1DConv(self.history_shape,
                                          cfg["history_channels"],
                                          cfg["history_fc_shape"],
                                          cfg["history_latent_size"],
                                          cfg["history_conv_kernel_size"],
                                          activation_fn=self.activation_fn)

        self.graph_map_encoder = MLP(cfg["graph_map_fc_shape"],
                                     self.activation_fn,
                                     self.node_m_shape[1],
                                     cfg["graph_map_latent_size"],
                                     init_scale=1.0 / np.sqrt(2),
                                     )

        if cfg["aggregator_type"] == "PN":
            self.aggregator = SimplePointNet([self.num_nodes, cfg["graph_map_latent_size"] + self.node_p_shape[1]],
                                            cfg["pointnet_channels"],
                                            cfg["pointnet_fc_shape"],
                                            cfg["aggregator_latent_size"],
                                            activation_fn=self.activation_fn)
        elif cfg["aggregator_type"] == "Attention":
            self.aggregator = MultiHeadAttention([self.num_nodes, cfg["graph_map_latent_size"] + self.node_p_shape[1]],
                                            cfg["aggregator_encoder_dim"],
                                            cfg["aggregator_encoder_dim"],
                                            cfg["aggregaor_num_heads"])
        elif cfg["aggregator_type"] == "DGCNN":
            self.aggregator = DGCNN(cfg["graph_map_latent_size"] + self.node_p_shape[1],
                                    cfg["aggregator_embedding_dim"],
                                    cfg["aggregator_k"],
                                    cfg["aggregaor_output"])


        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)

        # self.velodyne_scan_encoder = Pool1DConv(self.velodyne_scan_shape,
        #                                         cfg["velodyne_scan_channels"],
        #                                         cfg["velodyne_scan_fc_shape"],
        #                                         cfg["velodyne_scan_latent_size"],
        #                                         cfg["velodyne_scan_conv_kernel_size"],
        #                                         activation_fn=self.activation_fn)
        self.velodyne_scan_encoder = MLP(
            cfg["velodyne_scan_fc_shape"],
            self.activation_fn,
            self.velodyne_scan_obs_size,
            cfg["velodyne_scan_latent_size"],
            init_scale=1.0 / np.sqrt(2),
        )

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            + cfg["velodyne_scan_latent_size"]
            + cfg["aggregator_latent_size"]
            + cfg["history_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
    )


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
        obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)
        velodyne_scan, node_p, node_m, history = self.split_obs(obs)

        # scan
        velodyne_scan_latent = self.velodyne_scan_encoder(velodyne_scan)

        # history
        history_latent = self.history_encoder(history)

        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_m_feature = self.graph_map_encoder(node_m)

        # aggregation
        node_m_feature = node_m_feature.reshape(obs.shape[0], self.num_nodes, -1)
        node_p_feature = node_p.reshape(obs.shape[0], self.num_nodes, -1)
        concat_node_feature = torch.cat([node_m_feature, node_p_feature], dim=2)

        pointnet_feature = self.aggregator(concat_node_feature)

        concat_feature = torch.cat([velodyne_scan_latent, history_latent, pointnet_feature], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.velodyne_scan_obs_size, self.node_p_size, self.node_m_size,
                                 self.history_obs_size], dim=1)


class NavPolicyOnlyRaysPt(NavPolicyOnlyRays):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__(obs_shape_dict, action_size, cfg)

        self.command_size = self.obs_shape_dict["command"][0]
        self.num_obs = self.command_size + self.velodyne_scan_obs_size + self.node_p_size + self.node_m_size + self.history_obs_size
        self.num_obs_normalizer = self.command_size + self.velodyne_scan_obs_size
        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.command_size
            + cfg["velodyne_scan_latent_size"]
            + cfg["aggregator_latent_size"]
            + cfg["history_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
        obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)
        commands, velodyne_scan, node_p, node_m, history = self.split_obs(obs)

        # scan
        velodyne_scan_latent = self.velodyne_scan_encoder(velodyne_scan)

        # history
        history_latent = self.history_encoder(history)

        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_m_feature = self.graph_map_encoder(node_m)

        # aggregation
        node_m_feature = node_m_feature.reshape(obs.shape[0], self.num_nodes, -1)
        node_p_feature = node_p.reshape(obs.shape[0], self.num_nodes, -1)
        concat_node_feature = torch.cat([node_m_feature, node_p_feature], dim=2)

        pointnet_feature = self.aggregator(concat_node_feature)

        concat_feature = torch.cat([commands, velodyne_scan_latent, history_latent, pointnet_feature], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.command_size, self.velodyne_scan_obs_size, self.node_p_size, self.node_m_size,
                                 self.history_obs_size], dim=1)




class NavPolicyRayLatent(nn.Module):

    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.obs_shape_dict = obs_shape_dict

        self.node_p_shape = self.obs_shape_dict["node_p"]
        self.node_m_shape = self.obs_shape_dict["node_m"]
        self.history_shape = self.obs_shape_dict["history"]
        self.scan_latent_shape = self.obs_shape_dict["scan_latent"]

        self.node_p_size = self.node_p_shape.numel()
        self.node_m_size = self.node_m_shape.numel()
        self.history_obs_size = self.history_shape.numel()
        self.scan_latent_size = self.scan_latent_shape.numel()

        self.num_nodes = self.node_p_shape[0]

        self.num_obs = self.scan_latent_size + self.node_p_size + self.node_m_size + self.history_obs_size
        self.action_size = action_size

        # self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)

        # Load configs
        self.activation_fn = get_activation(cfg["activation_fn"])


        self.history_encoder = Pool1DConv(self.history_shape,
                                          cfg["history_channels"],
                                          cfg["history_fc_shape"],
                                          cfg["history_latent_size"],
                                          cfg["history_conv_kernel_size"],
                                          activation_fn=self.activation_fn)


        self.graph_feature_encoder = MLP(cfg["graph_feature_fc_shape"],
                                         self.activation_fn,
                                         self.node_m_shape[1] + self.node_p_shape[1],
                                         cfg["graph_feature_latent_size"],
                                         init_scale=1.0 / np.sqrt(2),
                                         )

        if cfg["aggregator_type"] == "PN":
            self.aggregator = SimplePointNet([self.num_nodes, cfg["graph_feature_latent_size"]],
                                            cfg["pointnet_channels"],
                                            cfg["pointnet_fc_shape"],
                                            cfg["aggregator_latent_size"],
                                            activation_fn=self.activation_fn)
        elif cfg["aggregator_type"] == "Attention":
            self.aggregator = MultiHeadAttention([self.num_nodes,cfg["graph_feature_latent_size"]],
                                            cfg["aggregator_encoder_dim"],
                                            cfg["aggregator_encoder_dim"],
                                            cfg["aggregaor_num_heads"])
        elif cfg["aggregator_type"] == "DGCNN":
            self.aggregator = DGCNN(cfg["graph_feature_latent_size"],
                                    cfg["aggregator_embedding_dim"],
                                    cfg["aggregator_k"],
                                    cfg["aggregaor_output"])

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.scan_latent_size
            + cfg["aggregator_latent_size"]
            + cfg["history_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
    )


    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        scan_latent, node_p, node_m, history = self.split_obs(obs)

        # history
        history_latent = self.history_encoder(history)

        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_p = node_p.reshape(obs.shape[0] * self.num_nodes, -1)
        concat_node_feature = torch.cat([node_m, node_p], dim=1)
        node_feature = self.graph_feature_encoder(concat_node_feature)
        node_feature = node_feature.reshape(obs.shape[0], self.num_nodes, -1)

        pointnet_feature = self.aggregator(node_feature)

        concat_feature = torch.cat([scan_latent, history_latent, pointnet_feature], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.scan_latent_size, self.node_p_size, self.node_m_size,
                                 self.history_obs_size], dim=1)


class NavPolicGoalRayLatent(nn.Module):

    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.obs_shape_dict = obs_shape_dict

        self.node_p_shape = self.obs_shape_dict["node_p"]
        self.node_m_shape = self.obs_shape_dict["node_m"]
        self.history_shape = self.obs_shape_dict["history"]
        self.scan_latent_shape = self.obs_shape_dict["scan_latent"]
        self.command_size = self.obs_shape_dict["command"][0]

        self.node_p_size = self.node_p_shape.numel()
        self.node_m_size = self.node_m_shape.numel()
        self.history_obs_size = self.history_shape.numel()
        self.scan_latent_size = self.scan_latent_shape.numel()

        self.num_nodes = self.node_p_shape[0]

        self.num_obs = self.scan_latent_size + self.node_p_size + self.node_m_size + self.history_obs_size + self.command_size
        self.action_size = action_size

        self.scan_normalizer = self.scan_latent_size
        self.scan_normalizer = EmpiricalNormalization(shape=[self.scan_latent_size], until=1.0e8)

        # Load configs
        self.activation_fn = get_activation(cfg["activation_fn"])

        self.history_encoder = Pool1DConv(self.history_shape,
                                          cfg["history_channels"],
                                          cfg["history_fc_shape"],
                                          cfg["history_latent_size"],
                                          cfg["history_conv_kernel_size"],
                                          activation_fn=self.activation_fn)

        self.graph_feature_encoder = MLP(cfg["graph_feature_fc_shape"],
                                         self.activation_fn,
                                         self.node_m_shape[1] + self.node_p_shape[1],
                                         cfg["graph_feature_latent_size"],
                                         init_scale=1.0 / np.sqrt(2),
                                         )

        if cfg["aggregator_type"] == "PN":
            self.aggregator = SimplePointNet([self.num_nodes, cfg["graph_feature_latent_size"]],
                                             cfg["pointnet_channels"],
                                             cfg["pointnet_fc_shape"],
                                             cfg["aggregator_latent_size"],
                                             activation_fn=self.activation_fn)
        elif cfg["aggregator_type"] == "Attention":
            self.aggregator = MultiHeadAttention([self.num_nodes, cfg["graph_feature_latent_size"]],
                                                 cfg["aggregator_encoder_dim"],
                                                 cfg["aggregator_encoder_dim"],
                                                 cfg["aggregaor_num_heads"])
        elif cfg["aggregator_type"] == "DGCNN":
            self.aggregator = DGCNN(cfg["graph_feature_latent_size"],
                                    cfg["aggregator_embedding_dim"],
                                    cfg["aggregator_k"],
                                    cfg["aggregaor_output"])

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.command_size + self.scan_latent_size
            + cfg["aggregator_latent_size"]
            + cfg["history_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        command, scan_latent, node_p, node_m, history = self.split_obs(obs)

        scan_latent = self.scan_normalizer(scan_latent)


        # history
        history_latent = self.history_encoder(history)

        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_m = self.scan_normalizer(node_m)

        node_p = node_p.reshape(obs.shape[0] * self.num_nodes, -1)

        concat_node_feature = torch.cat([node_m, node_p], dim=1)
        node_feature = self.graph_feature_encoder(concat_node_feature)
        node_feature = node_feature.reshape(obs.shape[0], self.num_nodes, -1)

        pointnet_feature = self.aggregator(node_feature)

        concat_feature = torch.cat([command, scan_latent, history_latent, pointnet_feature], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.command_size, self.scan_latent_size, self.node_p_size, self.node_m_size,
                                 self.history_obs_size], dim=1)


class NavPolicGoalRayLatent2(nn.Module):

    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.obs_shape_dict = obs_shape_dict

        self.command_size = self.obs_shape_dict["command"][0]

        self.node_p_shape = self.obs_shape_dict["node_p"]
        self.node_m_shape = self.obs_shape_dict["node_m"]
        self.history_shape = self.obs_shape_dict["history"]
        self.scan_latent_shape = self.obs_shape_dict["scan_latent"]
        self.near_nodes_shape = self.obs_shape_dict["near_nodes"]


        self.node_p_size = self.node_p_shape.numel()
        self.node_m_size = self.node_m_shape.numel()
        self.history_obs_size = self.history_shape.numel()
        self.scan_latent_size = self.scan_latent_shape.numel()
        self.near_nodes_size = self.near_nodes_shape.numel()

        self.num_nodes = self.node_p_shape[0]

        self.num_obs = self.scan_latent_size + self.node_p_size + self.node_m_size + self.history_obs_size + self.command_size + self.near_nodes_size
        self.action_size = action_size

        self.scan_normalizer = self.scan_latent_size
        self.scan_normalizer = EmpiricalNormalization(shape=[self.scan_latent_size], until=1.0e8)

        # Load configs
        self.activation_fn = get_activation(cfg["activation_fn"])

        self.history_encoder = Pool1DConv(self.history_shape,
                                          cfg["history_channels"],
                                          cfg["history_fc_shape"],
                                          cfg["history_latent_size"],
                                          cfg["history_conv_kernel_size"],
                                          activation_fn=self.activation_fn)

        self.graph_feature_encoder = MLP(cfg["graph_feature_fc_shape"],
                                         self.activation_fn,
                                         self.node_m_shape[1] + self.node_p_shape[1] + self.command_size,
                                         # self.node_m_shape[1] + self.node_p_shape[1],
                                         cfg["graph_feature_latent_size"],
                                         init_scale=1.0 / np.sqrt(2),
                                         )
        # TODO: goal info into graph_feature_encoder

        if cfg["aggregator_type"] == "PN":
            self.aggregator = SimplePointNet([self.num_nodes, cfg["graph_feature_latent_size"]],
                                             cfg["pointnet_channels"],
                                             cfg["pointnet_fc_shape"],
                                             cfg["aggregator_latent_size"],
                                             activation_fn=self.activation_fn)
        elif cfg["aggregator_type"] == "Attention":
            self.aggregator = MultiHeadAttention([self.num_nodes, cfg["graph_feature_latent_size"]],
                                                 cfg["aggregator_encoder_dim"],
                                                 cfg["aggregator_encoder_dim"],
                                                 cfg["aggregaor_num_heads"])
        elif cfg["aggregator_type"] == "DGCNN":
            self.aggregator = DGCNN(cfg["graph_feature_latent_size"],
                                    cfg["aggregator_embedding_dim"],
                                    cfg["aggregator_k"],
                                    cfg["aggregaor_output"])

        self.near_nodes_encoder = SimplePointNet([self.near_nodes_shape[0], self.near_nodes_shape[1]],
                                                    cfg["near_nodes_channels"],
                                                    cfg["near_nodes_fc_shape"],
                                                    cfg["near_nodes_latent_size"],
                                                    activation_fn=self.activation_fn)
        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            self.command_size + self.scan_latent_size
            + cfg["aggregator_latent_size"]
            + cfg["history_latent_size"]
            + cfg["near_nodes_latent_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        command, scan_latent, node_p, node_m, history, near_nodes = self.split_obs(obs)

        scan_latent = self.scan_normalizer(scan_latent)

        # history
        history_latent = self.history_encoder(history)

        # nearby nodes
        near_nodes_latent = self.near_nodes_encoder(near_nodes)

        # graph_map
        node_m = node_m.reshape(node_m.shape[0] * self.num_nodes, -1)
        node_m = self.scan_normalizer(node_m)

        # TODO: append command
        command_repeated = command.repeat(1, self.num_nodes).reshape(-1, self.num_nodes, command.shape[1])

        node_p = node_p.reshape(node_p.shape[0], self.num_nodes, -1)
        node_p = torch.cat([node_p, command_repeated], dim=2) # shape = (batch_size, num_nodes, node_p_size + command_size)

        node_p = node_p.reshape(obs.shape[0] * self.num_nodes, -1)

        concat_node_feature = torch.cat([node_m, node_p], dim=1)
        node_feature = self.graph_feature_encoder(concat_node_feature)
        node_feature = node_feature.reshape(obs.shape[0], self.num_nodes, -1)

        pointnet_feature = self.aggregator(node_feature)

        concat_feature = torch.cat([command, scan_latent, history_latent, pointnet_feature, near_nodes_latent], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.command_size, self.scan_latent_size, self.node_p_size, self.node_m_size,
                                 self.history_obs_size, self.near_nodes_size], dim=1)


class EstimatorBase(nn.Module):
    def __init__(self, obs_shape_dict, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.obs_shape_dict = obs_shape_dict

        self.map_shape = self.obs_shape_dict["bev"]
        self.goal_size = self.obs_shape_dict["goal"][0]
        print("check")
        print(self.map_shape)

        if isinstance(self.map_shape, list):
            self.map_size = np.prod(np.array(self.map_shape))
        else:
            self.map_size = self.map_shape.numel()

        self.num_obs = self.map_size + self.goal_size

        self.map_normalizer = EmpiricalNormalization(shape=[self.map_size], until=1.0e5)

        # Load configs
        self.activation_fn = get_activation(cfg.activation_fn)

        self.map_encoder = SimpleCNN2(self.map_shape,
                                     cfg.cnn_channels,
                                     cfg.cnn_fc_shape,
                                     cfg.map_latent_size,
                                     activation_fn=self.activation_fn,
                                     kernel_size=3,
                                     downsample_stride=2)  # TODO: later maybe


        self.output_mlp = MLP(
            cfg.output_mlp_size,
            self.activation_fn,
            cfg.map_latent_size + self.goal_size,
            1,
            init_scale=1.0 / np.sqrt(2),
        )

        # self.output_mlp2 = MLP(
        #     cfg.output_mlp_size,
        #     self.activation_fn,
        #     cfg.map_latent_size + self.goal_size,
        #     6, # xyz + rpy
        #     init_scale=1.0 / np.sqrt(2),
        # )

    def forward(self, map: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:

        # scan
        map_latent = self.map_encoder(map)


        concat_feature = torch.cat([map_latent, goal], dim=1)
        output = self.output_mlp(concat_feature)
        output = torch.sigmoid(output)

        # output2 = self.output_mlp2(concat_feature)
        #
        # output_concat = torch.cat([output, output2], dim=1)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.map_size, self.goal_size], dim=1)

class EstimatorTwoInputs(nn.Module):
    def __init__(self, obs_shape_dict, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.obs_shape_dict = obs_shape_dict

        self.ray_size = self.obs_shape_dict["rays"][0]
        self.goal_size = self.obs_shape_dict["goal"][0]
        self.map_shape = self.obs_shape_dict["bev"]

        if isinstance(self.map_shape, list):
            self.map_size = np.prod(np.array(self.map_shape))
        else:
            self.map_size = self.map_shape.numel()


        # Load configs
        self.activation_fn = get_activation(cfg.activation_fn)

        self.bev_encoder = SimpleCNN2(self.map_shape,
                                     cfg.cnn_channels,
                                     cfg.cnn_fc_shape,
                                     cfg.map_latent_size,
                                     activation_fn=self.activation_fn,
                                     kernel_size=3,
                                     downsample_stride=2)  # TODO: later maybe


        self.num_obs = self.ray_size + self.goal_size + self.map_size

        self.map_normalizer = EmpiricalNormalization(shape=[self.ray_size], until=1.0e5)

        # Load configs
        self.activation_fn = get_activation(cfg.activation_fn)

        self.ray_encoder = MLP(cfg.ray_cast_fc_shape,
                                self.activation_fn,
                                self.ray_size,
                                cfg.ray_cast_latent_size,
                                init_scale=1.0 / np.sqrt(2))


        self.output_mlp = MLP(
            cfg.output_mlp_size,
            self.activation_fn,
            cfg.map_latent_size + cfg.ray_cast_latent_size + self.goal_size,
            1,
            init_scale=1.0 / np.sqrt(2),
        )
        self.output_mlp2 = MLP(
            cfg.output_mlp_size,
            self.activation_fn,
            cfg.map_latent_size  + cfg.ray_cast_latent_size + self.goal_size,
            676,
            init_scale=1.0 / np.sqrt(2),
        )

    def forward(self, bev, ray: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:

        map_latent = self.bev_encoder(bev)
        ray_latent = self.ray_encoder(ray) * 0.0

        concat_feature = torch.cat([ray_latent, map_latent, goal], dim=1)
        output = self.output_mlp(concat_feature)
        output = torch.sigmoid(output)

        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.ray_size, self.goal_size], dim=1)


class RayCastEncoder(nn.Module):
    def __init__(self, input_size, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.ray_size = input_size

        # Load configs
        self.activation_fn = get_activation(cfg.activation_fn)

        print("ENCODER MLP SIZE", cfg.ray_encoder_mlp_size)
        print("RAY SIZE", self.ray_size)
        print("RAY CAST LATENT SIZE", cfg.ray_cast_latent_size)
        self.ray_encoder = MLP(cfg.ray_encoder_mlp_size,
                                self.activation_fn,
                                self.ray_size,
                                cfg.ray_cast_latent_size,
                                init_scale=1.0 / np.sqrt(2))


    def forward(self, ray: torch.Tensor) -> torch.Tensor:
        ray_latent = self.ray_encoder(ray)
        return ray_latent


class RayCastVEncoder(nn.Module):
    def __init__(self, input_size, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.ray_size = input_size

        # Load configs
        self.activation_fn = get_activation(cfg.activation_fn)

        print("ENCODER MLP SIZE", cfg.ray_encoder_mlp_size)
        print("RAY SIZE", self.ray_size)
        print("RAY CAST LATENT SIZE", cfg.ray_cast_latent_size)
        self.ray_encoder = MLP(cfg.ray_encoder_mlp_size,
                                self.activation_fn,
                                self.ray_size,
                                cfg.ray_cast_latent_size  * 2,
                                init_scale=1.0 / np.sqrt(2))

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, ray: torch.Tensor):
        ray_latent = self.ray_encoder(ray)

        mean, log_var = torch.chunk(ray_latent, 2, dim=1)

        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        return z, mean, log_var



class RayCastDecoder(nn.Module):
    def __init__(self, obs_shape_dict, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.obs_shape_dict = obs_shape_dict

        self.ray_size = self.obs_shape_dict["rays"][0]
        self.goal_size = self.obs_shape_dict["goal"][0]
        self.map_shape = self.obs_shape_dict["bev"]

        if isinstance(self.map_shape, list):
            self.map_size = np.prod(np.array(self.map_shape))
        else:
            self.map_size = self.map_shape.numel()

        self.input_dim = cfg.ray_cast_latent_size

        # Load configs
        self.activation_fn = get_activation(cfg.activation_fn)
        self.num_obs = self.ray_size

        # Load configs
        self.activation_fn = get_activation(cfg.activation_fn)

        self.output_mlp_ray = MLP(
            cfg.decoder_mlp_size_ray,
            self.activation_fn,
            cfg.ray_cast_latent_size,
            self.ray_size,
        )

        self.output_mlp_map = MLP(
            cfg.decoder_mlp_size_map,
            self.activation_fn,
            cfg.ray_cast_latent_size,
            self.map_size,
        )

    def get_rays(self, ray_latent) -> torch.Tensor:
        output = self.output_mlp_ray(ray_latent)
        return output

    def get_map(self, ray_latent) -> torch.Tensor:
        output = self.output_mlp_map(ray_latent)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.ray_size, self.goal_size], dim=1)



class GridmapEncoder(nn.Module):
    def __init__(self, map_shape, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.map_shape = map_shape

        print("check map shape")
        print(self.map_shape)

        if isinstance(self.map_shape, list):
            self.map_size = np.prod(np.array(self.map_shape))
        else:
            self.map_size = self.map_shape.numel()

        # Load configs
        self.activation_fn = get_activation(cfg.activation_fn)

        self.map_encoder = SimpleCNN2(self.map_shape,
                                     cfg.map_encoder_cnn_channels,
                                     cfg.map_encoder_fc_shape,
                                     cfg.map_latent_size,
                                     activation_fn=self.activation_fn,
                                     kernel_size=3,
                                     downsample_stride=2)  # TODO: later maybe

        # self.map_encoder = SimpleCNN(self.map_shape,
        #                              cfg.map_encoder_cnn_channels,
        #                              cfg.map_encoder_fc_shape,
        #                              cfg.map_latent_size,
        #                              activation_fn=self.activation_fn)

        # self.map_encoder = MLP(cfg.map_encoder_fc_shape,
        #                        self.activation_fn,
        #                        self.map_size,
        #                        cfg.map_latent_size,
        #                        )


    def forward(self, map: torch.Tensor) -> torch.Tensor:
        # scan
        map_latent = self.map_encoder(map)
        return map_latent



class GridmapDecoder(nn.Module):
    def __init__(self, input_size, map_shape, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.map_shape = map_shape

        print("check map shape")
        print(self.map_shape)

        if isinstance(self.map_shape, list):
            self.map_size = np.prod(np.array(self.map_shape))
        else:
            self.map_size = self.map_shape.numel()

        # Load configs
        self.activation_fn = get_activation(cfg.activation_fn)

        self.map_decoder = MLP(cfg.map_decoder_mlp_size,
                                 self.activation_fn,
                                    self.input_size,
                                    self.map_size,
                                    init_scale=1.0 / np.sqrt(2))


    def forward(self,latent) -> torch.Tensor:
        # scan
        map_recon = self.map_decoder(latent)
        map_recon = map_recon.view(-1, *self.map_shape)

        return map_recon



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
