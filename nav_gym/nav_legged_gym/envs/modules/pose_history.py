import networkx as nx


# TODO: use nx
# TODO: remove "1000" trick
from typing import Optional, Sequence, List

from copy import copy, deepcopy
import numpy as np
import torch
import os

# utils
from isaacgym.torch_utils import to_torch, get_axis_params, quat_rotate_inverse
from nav_gym.nav_legged_gym.utils.math_utils import quat_apply_yaw, quat_conjugate, quat_mul, get_euler_xyz, yaw_quat


class PoseHistoryData:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = -1

        self.num_poses = None
        self.history_pos_abs = None
        self.history_pos = None
        self.history_quat_abs = None
        self.history_yaw = None
        self.use_orientation = False

    def init_buffers(self, num_envs, use_orientation, device):
        self.use_orientation = use_orientation

        self.device = device
        self.num_envs = num_envs

        self.num_poses = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.history_pos_abs = torch.zeros(
            (self.num_envs, self.cfg.wp_history_len, 3), device=self.device, dtype=torch.float
        )

        self.history_pos = torch.zeros(
            (self.num_envs, self.cfg.wp_history_len, 3), device=self.device, dtype=torch.float
        )

        if self.use_orientation:
            self.history_quat_abs = torch.zeros(
                (self.num_envs, self.cfg.wp_history_len, 4), device=self.device, dtype=torch.float
            )
            self.history_rpy = torch.zeros(
                (self.num_envs, self.cfg.wp_history_len, 3), device=self.device, dtype=torch.float
            )
        else:
            self.history_quat_abs = None
            self.history_rpy = None

        self.inds = (
            torch.arange(self.cfg.wp_history_len).unsqueeze(0).repeat(self.num_envs, 1).to(self.device)
        )  # used in reward
        self.num_poses[:] = 0

    def reset_buffers(self, env, env_ids: Optional[Sequence[int]] = None):
        self.history_pos_abs[env_ids, :, :] = 0.0
        self.history_pos[env_ids, :, :] = 0.0
        if self.use_orientation:
            self.history_quat_abs[env_ids, :, :] = 0.0
            self.history_yaw[env_ids, :, :] = 0.0

        self.num_poses[env_ids] = 0

    def add_pose(self, position, orientation, env_ids=None):
        if env_ids is None:
            env_ids = ...  # all elements of the tensor

        temp = self.history_pos_abs[
            env_ids, :-1, :
        ].clone()  # JL: I dont know if this is necessary. In cpp we need this :p
        self.history_pos_abs[env_ids, 1:, :] = temp
        self.history_pos_abs[env_ids, 0, :] = position[env_ids]

        if self.use_orientation:
            temp = self.history_quat_abs[env_ids, :-1, :].clone()
            self.history_quat_abs[env_ids, 1:, :] = temp
            self.history_quat_abs[env_ids, 0, :] = orientation[env_ids]

        self.num_poses[env_ids] += 1
        self.num_poses = torch.clamp(self.num_poses, 0, self.cfg.wp_history_len)

    def update_buffers(self, env, quats=None):
        if quats is None:
            quats = env.robot.root_quat_w

        dis = env.robot.root_pos_w.unsqueeze(1) - self.history_pos_abs # TODO: fix this

        # if env.cfg.only_yaw:
        #     quats = yaw_quat(quats)
        dis_base = quat_rotate_inverse(
            quats.unsqueeze(1).repeat(1, self.history_pos_abs.shape[1], 1).reshape(-1, 4),
            dis.reshape(-1, 3),
        )
        self.history_pos = dis_base.reshape(self.num_envs, self.history_pos_abs.shape[1], 3)

        # set zero for the poses that are not used
        self.history_pos = self.history_pos * (self.inds < self.num_poses.unsqueeze(1)).unsqueeze(2)

        if self.use_orientation:
            rolls, pitches, yaws = get_euler_xyz(self.history_quat_abs.shape[1], 1).reshape(-1, 4)
            self.history_rpy[:, :, 0] = rolls.reshape(self.num_envs, self.history_quat_abs.shape[1])
            self.history_rpy[:, :, 1] = pitches.reshape(self.num_envs, self.history_quat_abs.shape[1])
            self.history_rpy[:, :, 2] = yaws.reshape(self.num_envs, self.history_quat_abs.shape[1])

            _, _, current_yaw = get_euler_xyz(quats)
            self.history_rpy[:, :, 2] = self.history_rpy[:, :, 2] - current_yaw.unsqueeze(1)

    # quats_repeated = quats.unsqueeze(1).repeat(1, self.history_quat_abs.shape[1], 1).reshape(-1, 4)
    # inverse_quat = quat_conjugate(quats_repeated)
    # abs_quat = self.history_quat_abs.reshape(-1, 4)
    # dis_quat = quat_mul(abs_quat, inverse_quat)
    #
    # rolls, pitches, yaws = get_euler_xyz(dis_quat)
    #
    #
    # # absolute roll pitch & yaw difference
    #
    # self.history_rpy[:, :, 0] = rolls.reshape(self.num_envs, self.history_quat_abs.shape[1])
    # self.history_rpy[:, :, 1] = pitches.reshape(self.num_envs, self.history_quat_abs.shape[1])
    # self.history_rpy[:, :, 2] = yaws.reshape(self.num_envs, self.history_quat_abs.shape[1])

    # TODO: yaw/orientation update
