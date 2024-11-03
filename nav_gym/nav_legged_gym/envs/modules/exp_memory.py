import networkx as nx


# TODO: use nx
from typing import Optional, Sequence, List

from copy import copy, deepcopy
import numpy as np
import torch
import os

# utils
from isaacgym.torch_utils import to_torch, get_axis_params, quat_rotate_inverse, quat_conjugate, quat_mul, get_euler_xyz
from nav_gym.nav_legged_gym.utils.math_utils import quat_apply_yaw


class ExplicitMemory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_node_cnt = 0

        self.use_local_map = self.cfg.use_local_map

        self.add_wp = None

        self.num_nodes = None
        self.all_graph_nodes_abs = None
        self.all_graph_nodes_abs_counts = None
        self.all_graph_nodes_quat_abs = None
        self.all_graph_nodes_quat = None
        self.all_graph_nodes = None
        self.all_graph_features = None

        self._comparator_func = self.cfg.comparator.func

    def init_buffers(self, num_envs, max_node_cnt, local_map_shape, device):
        self.device = device
        self.max_node_cnt = max_node_cnt
        self.num_envs = num_envs

        self.add_wp = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.num_nodes = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.all_graph_nodes_abs = torch.ones(
            (self.num_envs, self.max_node_cnt, 3), device=self.device, dtype=torch.float
        )
        self.all_graph_nodes_abs_counts = torch.zeros(
            (self.num_envs, self.max_node_cnt), device=self.device, dtype=torch.int
        )

        self.all_graph_nodes = torch.zeros((self.num_envs, self.max_node_cnt, 3), device=self.device, dtype=torch.float)
        self.all_graph_nodes_quat_abs = torch.ones(
            (self.num_envs, self.max_node_cnt, 4), device=self.device, dtype=torch.float
        )
        self.all_graph_nodes_quat_abs[:, :, 3] = 1.0

        self.all_graph_nodes_quat = torch.ones(
            (self.num_envs, self.max_node_cnt, 4), device=self.device, dtype=torch.float
        )
        self.all_graph_nodes_quat[:, :, 3] = 1.0

        if self.use_local_map:
            self.local_map_shape = [*local_map_shape]  # TODO: add channels if 2D or 3D data
            print("local_map_shape", local_map_shape)
            print("local_map_shape", self.local_map_shape)

            # input_dim = self.local_map_cfg.feature_dim**2
            self.all_graph_features = torch.zeros(
                (self.num_envs, self.max_node_cnt, *self.local_map_shape), device=self.device, dtype=torch.float
            )
        else:
            self.local_map_shape = None
            self.all_graph_features = None

    def reset_buffers(self, env, env_ids: Optional[Sequence[int]] = None):
        self.num_nodes[env_ids] = 0
        self.all_graph_nodes_abs[env_ids, :, :] = 0.0
        self.all_graph_nodes_abs_counts[env_ids, :] = 0
        self.all_graph_nodes_quat_abs[env_ids, :, :] = 0.0
        self.all_graph_nodes_quat_abs[env_ids, :, 3] = 1.0

        self.all_graph_nodes_quat[env_ids, :, :] = 0.0
        self.all_graph_nodes_quat[env_ids, :, 3] = 1.0

        if self.use_local_map and self.local_map_shape is not None:
            self.all_graph_features[env_ids, :, :] = 0.0

    def check_position(self, env, env_ids, positions):
        """
        check if position is already in graph
        """
        self.add_wp[:] = False

        # check distance to previously added nodes
        self.add_wp[env_ids] = self._comparator_func(self, self.cfg.comparator.params, env_ids, positions, env)

    def add_position_and_feature(self, env, env_ids, positions, feature, quats):
        """
        add waypoint computed by policy to global map, with certain comparator function
         -reach_nodes: future node poses, length = num_envs
        """

        self.check_position(env, env_ids, positions)
        # Indices of envs to be updated
        add_env_ids = self.add_wp.nonzero()[:, 0]

        # save position and node id
        graph_nodes_abs_temp = self.all_graph_nodes_abs[add_env_ids, :-1, :].clone()
        self.all_graph_nodes_abs[add_env_ids, 1:, :] = graph_nodes_abs_temp
        self.all_graph_nodes_abs[add_env_ids, 0, :] = positions[self.add_wp]

        all_graph_nodes_abs_counts_temp = self.all_graph_nodes_abs_counts[add_env_ids, :-1].clone()
        self.all_graph_nodes_abs_counts[add_env_ids, 1:] = all_graph_nodes_abs_counts_temp
        self.all_graph_nodes_abs_counts[add_env_ids, 0] = 0

        quats_temp = self.all_graph_nodes_quat_abs[add_env_ids, :-1, :].clone()
        self.all_graph_nodes_quat_abs[add_env_ids, 1:, :] = quats_temp
        self.all_graph_nodes_quat_abs[add_env_ids, 0, :] = quats[self.add_wp]

        # add feature
        if len(add_env_ids) > 0 and self.use_local_map:

            feature_to_add = feature[add_env_ids]
            all_graph_features_temp = self.all_graph_features[add_env_ids, :-1, :].clone()
            self.all_graph_features[add_env_ids, 1:, :] = all_graph_features_temp
            self.all_graph_features[add_env_ids, 0, :] = feature_to_add

        self.num_nodes[add_env_ids] += 1

    def update_buffers(self, env, env_ids: Optional[Sequence[int]] = None, quats: torch.Tensor = None):
        dis = env.robot.root_pos_w.unsqueeze(1) - self.all_graph_nodes_abs

        if quats is None:
            quats = env.robot.root_quat_w

        quats_repeated = quats.unsqueeze(1).repeat(1, self.max_node_cnt, 1).reshape(-1, 4)
        dis_base = quat_rotate_inverse(quats_repeated, dis.reshape(-1, 3))

        self.all_graph_nodes = dis_base.reshape(self.num_envs, self.max_node_cnt, 3).clone()

        if env_ids is None:
            env_ids = ...  # all elements of the tensor

        min_dis = torch.min(
            torch.norm(env.robot.root_pos_w[:, :2].unsqueeze(1) - self.all_graph_nodes_abs[:, :, :2], dim=2), dim=1
        )
        ids = torch.arange(self.num_envs, device=self.device) if env_ids == ... else env_ids

        self.all_graph_nodes_abs_counts[ids, min_dis[1][ids]] += 1  # TODO: depending on distance margin

        # relative orientations
        inverse_quat = quat_conjugate(quats_repeated)
        abs_quat = self.all_graph_nodes_quat_abs.reshape(-1, 4)

        dis_quat = quat_mul(abs_quat, inverse_quat)
        self.all_graph_nodes_quat = dis_quat.reshape(self.num_envs, self.max_node_cnt, 4)

        # DEBUG CHECK
        # _, _, root_yaws = get_euler_xyz(quats)
        # print("DEBUG_REL_ORI")
        # print("ROOT_YAW", root_yaws[0])
        # _, _, yaws = get_euler_xyz(self.all_graph_nodes_quat_abs.reshape(-1, 4))
        # print("ABS_YAW", yaws.reshape(self.num_envs, self.max_node_cnt)[0, :5])
        # rel_yaw_1 = yaws.reshape(self.num_envs, self.max_node_cnt) - root_yaws.unsqueeze(1)
        # print("REL_YAW1", rel_yaw_1[0, :5])
        # _, _, rel_yaw_2 = get_euler_xyz(dis_quat)
        # rel_yaw_2 = rel_yaw_2.reshape(self.num_envs, self.max_node_cnt)
        # print("REL_YAW2", rel_yaw_2[0, :5])
        # print("DIFF", rel_yaw_1[0, :5] - rel_yaw_2[0, :5])

        # self.root_lin_vel_b[env_ids] = quat_rotate_inverse(self.root_quat_w[env_ids], self.root_lin_vel_w[env_ids])
        # self.projected_gravity_b[env_ids] = quat_rotate_inverse(self.root_quat_w[env_ids], self._gravity_vec_w[env_ids])
