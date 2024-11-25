from isaacgym.torch_utils import (
    quat_rotate,
    quat_rotate_inverse,
    get_euler_xyz,
    quat_from_euler_xyz,
)


import torch
from nav_gym import NAV_GYM_ROOT_DIR
from typing import TYPE_CHECKING, Union
import os
from nav_gym.learning.datasets.motion_loader import MotionLoader
if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.envs.locomotion_mimic_env import LocomotionMimicEnv
    ANY_ENV = Union[LocomotionMimicEnv]
from collections import OrderedDict

class MimicModule:
    def __init__(self, env:"ANY_ENV"):
        self.num_envs = env.num_envs

        self.robot = env.robot
        self.base_pos_w = self.robot.root_pos_w
        self.base_quat_w = self.robot.root_quat_w
        self.base_lin_vel_w = self.robot.root_lin_vel_w
        self.base_ang_vel_w = self.robot.root_ang_vel_w
        self.projected_gravity_b = self.robot.projected_gravity_b
        self.dof_pos = self.robot.dof_pos
        self.default_dof_pos = self.robot.default_dof_pos
        self.dof_vel = self.robot.dof_vel
        #Load the Motion Data
        self.datasets_root = os.path.join(NAV_GYM_ROOT_DIR + "/resources/fld/motion_data/")
        self.motion_names = ["motion_data_pace1.0.pt","motion_data_walk01_0.5.pt","motion_data_walk03_0.5.pt","motion_data_canter02_1.5.pt"]

        self.motion_loader = MotionLoader(
            device="cuda",
            file_names=self.motion_names,
            file_root=self.datasets_root,
            corruption_level=0.0,
            reference_observation_horizon=2,
            test_mode=False,
            test_observation_dim=None
        )
        self.motion_idx = 0
        self.num_motion_clips, self.num_steps, self.motion_features_dim = self.motion_loader.data_list[self.motion_idx].size()
        self.cur_step = 0

    def on_env_post_physics_step(self):
        self._update()
    def _update(self):
        self.cur_step += 1
        if self.cur_step >= self.num_steps:
            self.cur_step = 0
#---------------------------------Getters---------------------------------
#Observations
    def get_dof_pos_leg_fr_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        #shape: (num_envs, num_dofs)
        return self.motion_loader.get_dof_pos_leg_fr(motion_data_per_step)
    def get_dof_pos_leg_fl_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        #shape: (num_envs, num_dofs)
        return self.motion_loader.get_dof_pos_leg_fl(motion_data_per_step)
    def get_dof_pos_leg_hr_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        #shape: (num_envs, num_dofs)
        return self.motion_loader.get_dof_pos_leg_hr(motion_data_per_step)
    def get_dof_pos_leg_hl_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        #shape: (num_envs, num_dofs)
        return self.motion_loader.get_dof_pos_leg_hl(motion_data_per_step)

    def get_dof_pos_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        #shape: (num_envs, num_dofs)
        return self.motion_loader.get_dof_pos(motion_data_per_step)
    def get_dof_vel_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        return self.motion_loader.get_dof_vel(motion_data_per_step)
    def get_base_lin_vel_w_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        return quat_rotate(self.base_quat_w, self.motion_loader.get_base_lin_vel_b(motion_data_per_step))
    def get_base_ang_vel_w_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        return quat_rotate(self.base_quat_w, self.motion_loader.get_base_ang_vel_b(motion_data_per_step))
