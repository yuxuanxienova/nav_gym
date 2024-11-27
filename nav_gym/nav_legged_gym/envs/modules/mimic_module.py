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

# class MimicModule:
#     def __init__(self, env:"ANY_ENV"):
#         self.num_envs = env.num_envs

#         self.robot = env.robot
#         self.base_pos_w = self.robot.root_pos_w
#         self.base_quat_w = self.robot.root_quat_w
#         self.base_lin_vel_w = self.robot.root_lin_vel_w
#         self.base_ang_vel_w = self.robot.root_ang_vel_w
#         self.projected_gravity_b = self.robot.projected_gravity_b
#         self.dof_pos = self.robot.dof_pos
#         self.default_dof_pos = self.robot.default_dof_pos
#         self.dof_vel = self.robot.dof_vel
#         #Load the Motion Data
#         self.datasets_root = os.path.join(NAV_GYM_ROOT_DIR + "/resources/fld/motion_data/")
#         self.motion_names = ["motion_data_pace1.0.pt","motion_data_walk01_0.5.pt","motion_data_walk03_0.5.pt","motion_data_canter02_1.5.pt"]

#         self.motion_loader = MotionLoader(
#             device="cuda",
#             file_names=self.motion_names,
#             file_root=self.datasets_root,
#             corruption_level=0.0,
#             reference_observation_horizon=2,
#             test_mode=False,
#             test_observation_dim=None
#         )
#         self.motion_idx = 0
#         self.num_motion_clips, self.num_steps, self.motion_features_dim = self.motion_loader.data_list[self.motion_idx].size()
#         self.cur_step = 0

#     def on_env_post_physics_step(self):
#         self._update()
#     def _update(self):
#         self.cur_step += 1
#         if self.cur_step >= self.num_steps:
#             self.cur_step = 0
# #---------------------------------Getters---------------------------------
# #Observations
#     def get_dof_pos_leg_fr_cur_step(self):
#         motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
#         #shape: (num_envs, num_dofs)
#         return self.motion_loader.get_dof_pos_leg_fr(motion_data_per_step)
#     def get_dof_pos_leg_fl_cur_step(self):
#         motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
#         #shape: (num_envs, num_dofs)
#         return self.motion_loader.get_dof_pos_leg_fl(motion_data_per_step)
#     def get_dof_pos_leg_hr_cur_step(self):
#         motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
#         #shape: (num_envs, num_dofs)
#         return self.motion_loader.get_dof_pos_leg_hr(motion_data_per_step)
#     def get_dof_pos_leg_hl_cur_step(self):
#         motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
#         #shape: (num_envs, num_dofs)
#         return self.motion_loader.get_dof_pos_leg_hl(motion_data_per_step)

#     def get_dof_pos_cur_step(self):
#         motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
#         #shape: (num_envs, num_dofs)
#         return self.motion_loader.get_dof_pos(motion_data_per_step)
#     def get_dof_vel_cur_step(self):
#         motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
#         return self.motion_loader.get_dof_vel(motion_data_per_step)
#     def get_base_lin_vel_w_cur_step(self):
#         motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
#         return quat_rotate(self.base_quat_w, self.motion_loader.get_base_lin_vel_b(motion_data_per_step))
#     def get_base_ang_vel_w_cur_step(self):
#         motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
#         return quat_rotate(self.base_quat_w, self.motion_loader.get_base_ang_vel_b(motion_data_per_step))
import os
import numpy as np
import matplotlib.pyplot as plt

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
        self.feet_pos_w = self.robot.feet_positions_w#(num_envs,num_feets,3)
        # Load the Motion Data
        self.datasets_root = os.path.join(NAV_GYM_ROOT_DIR + "/resources/fld/motion_data/")
        self.motion_names = ["motion_data_pace1.0_with_feet_pos.pt"]

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

        # Initialize data for plotting DOF positions
        self.dof_pos_robot_history = []
        self.dof_pos_motion_history = []
        self.time_steps = []

        # Initialize plotting
        self.plot_initialized = False
        self.num_dofs = self.dof_pos.shape[1]  # Assuming self.dof_pos shape is (num_envs, num_dofs)

    def on_env_post_physics_step(self):
        self._update()

    def _update(self):
        self.cur_step += 1
        if self.cur_step >= self.num_steps:
            self.cur_step = 0

        # Collect data for plotting
        robot_dof_pos = self.dof_pos.clone()
        desired_dof_pos = self.get_target_dof_pos_cur_step()

        # Select the first environment for plotting
        robot_dof_pos_env0 = robot_dof_pos[0].cpu().numpy()
        desired_dof_pos_env0 = desired_dof_pos[0].cpu().numpy()

        self.dof_pos_robot_history.append(robot_dof_pos_env0)
        self.dof_pos_motion_history.append(desired_dof_pos_env0)
        self.time_steps.append(self.cur_step)

        # Update the plot every N steps to reduce overhead
        N = 10  # Update plot every N steps
        if self.cur_step % N == 0:
            self.update_plot()

    # ---------------------------------Getters---------------------------------
    # Observations
    def get_robot_feet_pos_b_LF(self):
        robot_feet_pos_w_LF = self.feet_pos_w[:,self.motion_loader.feet_pos_name_to_id['LF_FOOT'],:]
        robot_feet_pos_b_LF = quat_rotate_inverse(self.base_quat_w, robot_feet_pos_w_LF - self.base_pos_w)
        return robot_feet_pos_b_LF
    
    def get_robot_feet_pos_b_LH(self):
        robot_feet_pos_w_LH = self.feet_pos_w[:,self.motion_loader.feet_pos_name_to_id['LH_FOOT'],:]
        robot_feet_pos_b_LH = quat_rotate_inverse(self.base_quat_w, robot_feet_pos_w_LH - self.base_pos_w)
        return robot_feet_pos_b_LH
    
    def get_robot_feet_pos_b_RF(self):
        robot_feet_pos_w_RF = self.feet_pos_w[:,self.motion_loader.feet_pos_name_to_id['RF_FOOT'],:]
        robot_feet_pos_b_RF = quat_rotate_inverse(self.base_quat_w, robot_feet_pos_w_RF - self.base_pos_w)
        return robot_feet_pos_b_RF
    
    def get_robot_feet_pos_b_RH(self):
        robot_feet_pos_w_RH = self.feet_pos_w[:,self.motion_loader.feet_pos_name_to_id['RH_FOOT'],:]
        robot_feet_pos_b_RH = quat_rotate_inverse(self.base_quat_w, robot_feet_pos_w_RH - self.base_pos_w)
        return robot_feet_pos_b_RH
    
    def get_target_feet_pos_b_LF_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        #return shape:(num_envs, 3)
        return self.motion_loader.get_feet_pos_b_LF(motion_data_per_step)
    
    def get_target_feet_pos_b_LH_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        #return shape:(num_envs, 3)
        return self.motion_loader.get_feet_pos_b_LH(motion_data_per_step)
    
    def get_target_feet_pos_b_RF_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        #return shape:(num_envs, 3)
        return self.motion_loader.get_feet_pos_b_RF(motion_data_per_step)
    
    def get_target_feet_pos_b_RH_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        #return shape:(num_envs, 3)
        return self.motion_loader.get_feet_pos_b_RH(motion_data_per_step)
    
    def get_target_dof_pos_leg_fr_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        # shape: (num_envs, num_dofs)
        return self.motion_loader.get_dof_pos_leg_fr(motion_data_per_step)

    def get_target_dof_pos_leg_fl_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        # shape: (num_envs, num_dofs)
        return self.motion_loader.get_dof_pos_leg_fl(motion_data_per_step)

    def get_target_dof_pos_leg_hr_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        # shape: (num_envs, num_dofs)
        return self.motion_loader.get_dof_pos_leg_hr(motion_data_per_step)

    def get_target_dof_pos_leg_hl_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        # shape: (num_envs, num_dofs)
        return self.motion_loader.get_dof_pos_leg_hl(motion_data_per_step)

    def get_target_dof_pos_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        # shape: (num_envs, num_dofs)
        return self.motion_loader.get_dof_pos(motion_data_per_step)

    def get_target_dof_vel_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        return self.motion_loader.get_dof_vel(motion_data_per_step)

    def get_target_base_lin_vel_w_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        return quat_rotate(self.base_quat_w, self.motion_loader.get_base_lin_vel_b(motion_data_per_step))

    def get_target_base_ang_vel_w_cur_step(self):
        motion_data_per_step = self.motion_loader.data_list[self.motion_idx][0, self.cur_step].repeat(self.num_envs, 1)
        return quat_rotate(self.base_quat_w, self.motion_loader.get_base_ang_vel_b(motion_data_per_step))

    # Method to initialize the plot
    def initialize_plot(self):
        plt.ion()  # Turn on interactive mode
        self.fig, self.axs = plt.subplots(self.num_dofs, 1, figsize=(10, 2 * self.num_dofs), sharex=True)
        if self.num_dofs == 1:
            self.axs = [self.axs]
        self.lines_robot = []
        self.lines_desired = []
        for i in range(self.num_dofs):
            line_robot, = self.axs[i].plot([], [], label='Robot DOF {}'.format(i))
            line_desired, = self.axs[i].plot([], [], label='Desired DOF {}'.format(i))
            self.axs[i].set_ylabel('DOF Position')
            self.axs[i].legend()
            self.lines_robot.append(line_robot)
            self.lines_desired.append(line_desired)
        self.axs[-1].set_xlabel('Time Step')
        self.plot_initialized = True

    # Method to update the plot
    def update_plot(self):
        if not self.plot_initialized:
            self.initialize_plot()

        M = 100  # Number of time steps to display (adjust as needed)
        dof_pos_robot_history = np.array(self.dof_pos_robot_history)  # Shape: (num_time_steps, num_dofs)
        dof_pos_motion_history = np.array(self.dof_pos_motion_history)
        time_steps = np.array(self.time_steps)

        # Limit data to the last M points to keep the plot clean
        if len(time_steps) > M:
            dof_pos_robot_history = dof_pos_robot_history[-M:]
            dof_pos_motion_history = dof_pos_motion_history[-M:]
            time_steps = time_steps[-M:]
            # Adjust time steps to be sequential for plotting
            time_steps = np.arange(len(time_steps))

        for i in range(self.num_dofs):
            self.lines_robot[i].set_data(time_steps, dof_pos_robot_history[:, i])
            self.lines_desired[i].set_data(time_steps, dof_pos_motion_history[:, i])
            self.axs[i].relim()
            self.axs[i].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()