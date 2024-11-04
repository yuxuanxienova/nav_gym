# isaac-gym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_apply
# python
from copy import deepcopy
import torch
import numpy as np
from typing import Tuple, Union, Dict, Any
import math
import torch
import abc
# legged-gym
from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
from nav_gym.nav_legged_gym.envs.config_locomotion_env import LocomotionEnvCfg
from nav_gym.nav_legged_gym.common.assets.robots.legged_robots.legged_robot import LeggedRobot
from nav_gym.nav_legged_gym.common.sensors.sensors import SensorBase, Raycaster
from nav_gym.nav_legged_gym.utils.math_utils import wrap_to_pi
from nav_gym.nav_legged_gym.common.terrain.terrain_unity import TerrainUnity
from nav_gym.nav_legged_gym.common.gym_interface import GymInterface
from nav_gym.nav_legged_gym.common.rewards.reward_manager import RewardManager
from nav_gym.nav_legged_gym.common.observations.observation_manager import ObsManager
from nav_gym.nav_legged_gym.common.terminations.termination_manager import TerminationManager
from nav_gym.nav_legged_gym.common.curriculum.curriculum_manager import CurriculumManager
from nav_gym.nav_legged_gym.common.sensors.sensor_manager import SensorManager
from nav_gym.nav_legged_gym.common.commands.command import CommandBase,UnifromVelocityCommand,UnifromVelocityCommandCfg
from nav_gym.nav_legged_gym.utils.visualization_utils import BatchWireframeSphereGeometry
from nav_gym.nav_legged_gym.envs.config_local_nav_env import LocalNavEnvCfg
import os
from nav_gym import NAV_GYM_ROOT_DIR
class LocalNavEnv:
    def __init__(self, cfg:LocalNavEnvCfg, ll_env_cls:LocomotionEnv) -> None:
        self.cfg = cfg
        #1. Parse the configuration
        cfg.ll_env_cfg.gym.headless = cfg.gym.headless
        cfg.ll_env_cfg.env.num_envs = cfg.env.num_envs
        #1.1 Create the low-level environment
        self.ll_env: LocomotionEnv = ll_env_cls(cfg.ll_env_cfg)
        self.ll_env.play_mode = True #enable play mode
        #1.2 Extract the necessary attributes
        self.sim = self.ll_env.sim
        self.gym = self.ll_env.gym
        self.gym_iface = self.ll_env.gym_iface
        self.num_envs = self.ll_env.num_envs
        self.device= self.ll_env.device
        self.robot = self.ll_env.robot
        self.terrain = self.ll_env.terrain
        self.viewer = self.ll_env.viewer
        self.num_actions = self.cfg.env.num_actions
        self.max_episode_length_s = self.cfg.env.episode_length_s # TODO 
        self.dt = self.ll_env.dt * self.cfg.hl_decimation
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.command_generator = self.ll_env.command_generator
        #1.3 Load the low-level policy
        scripted_model_path = os.path.join(NAV_GYM_ROOT_DIR, "resources/model/low_level/" )
        file_name = "ll_jit_model.pt"
        #Load the policy
        try:
            self.ll_policy = torch.jit.load(scripted_model_path + file_name).to("cuda:0")
            self.ll_policy.eval()
        except Exception as e:
            print("Loading scripted model failed:", e)
            exit(1)

        
        #2. Prepare mdp helper managers
        self._init_buffers()
        self.sensor_manager = SensorManager(self)
        self.reward_manager = RewardManager(self)
        self.obs_manager = ObsManager(self)
        self.termination_manager = TerminationManager(self)
        self.curriculum_manager = CurriculumManager(self)
        #3. Initialize the environment
        self.reset()
        self.obs_dict = self.obs_manager.compute_obs(self)
        #4. others
        self.num_obs = self.obs_manager.get_obs_dims_from_group("policy")
        self.num_privileged_obs = self.obs_manager.get_obs_dims_from_group("privileged")
    def _init_buffers(self):
        self.obs_dict = dict()
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.ll_reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = dict()

        self.pos_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_time_left = torch.zeros(self.num_envs, device=self.device)
        #Set the target position
        self.pos_target[:, 0] = self.ll_env.terrain.x_goal
        self.pos_target[:, 1] = self.ll_env.terrain.y_goal
        self.pos_target[:, 2] = 0.5

        # targets given to ll by hl x_vel, y_vel, yaw_vel
        self.command_x_vel = torch.zeros(self.num_envs, device=self.device)
        self.command_y_vel = torch.zeros(self.num_envs, device=self.device)
        self.command_yaw_vel = torch.zeros(self.num_envs, device=self.device)

        # metrics for losses
        self.total_sq_torques = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.total_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_pos = self.robot.root_pos_w.clone()

    def step(self, actions):
        self._preprocess_actions(actions)
        self.ll_reset_buf.zero_()
        for _ in range(self.cfg.hl_decimation):
            ll_actions = self._get_ll_actions()
            _,_, _, dones, _ = self.ll_env.step(ll_actions)
            self.ll_reset_buf |= dones
            #visualization
            self._draw_hl_debug_vis()
        # sensors need to be updated
        self.sensor_manager.update()
        self._post_ll_step()
        # print(self.ll_env.reward_manager.episode_sums["contact_forces"])
        return self.obs_dict["policy"], self.obs_manager.get_obs_from_group("privileged"),self.rew_buf, self.reset_buf, self.extras
    
    def _preprocess_actions(self, actions: torch.Tensor):
        self.ll_env.set_velocity_commands(actions)
        return actions  
    def _get_ll_actions(self):
        """Apply actions to simulation buffers in the environment."""
        self.ll_env._update_commands()
        obs_ll = self.ll_env.obs_manager.compute_obs(self.ll_env)["policy"]
        ll_action = self.ll_policy(obs_ll)
        return ll_action  
    def _draw_hl_debug_vis(self):
        if self.cfg.env.enable_debug_vis:
            self.sensor_manager.debug_vis(self.ll_env.envs)
    def _post_ll_step(self):
        #update the metrics
        self.command_time_left -= self.dt
        self.total_distance += torch.linalg.norm(self.robot.root_pos_w - self.previous_pos, dim=-1)
        self.previous_pos[:] = self.robot.root_pos_w

        self.reset_buf[:] = self.termination_manager.check_termination(self)
        self.rew_buf[:] = self.reward_manager.compute_reward(self)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) != 0:
            time_outs = self.termination_manager.time_out_buf.nonzero(as_tuple=False).flatten()
            self.extras["episode"] = {"dist_to_goal": torch.mean(torch.norm(self.pos_target[env_ids] - self.robot.root_pos_w[env_ids], dim=1)),
                                      "death_rate": (len(env_ids) - len(time_outs))/len(env_ids),
                                      "dist_to_goal_timeout": torch.mean(torch.norm(self.pos_target[time_outs] - self.robot.root_pos_w[time_outs], dim=1)), 
                                      "total_distance": torch.mean(self.total_distance[env_ids])}
            self.reward_manager.log_info(self, env_ids, self.extras["episode"])
            self.reset_idx(env_ids)

        self.obs_dict = self.obs_manager.compute_obs(self)
    def reset(self):
        """Reset all environment instances."""
        # reset environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.obs_dict = self.obs_manager.compute_obs(self)
        self.obs_buf = self.obs_dict["policy"]
        self.extras["observations"] = self.obs_dict
        # return obs
        return self.obs_buf, self.extras
    def reset_idx(self,env_ids):
        self.ll_env.reset_idx(env_ids)
        self.sensor_manager.update()

        # self.extras["episode"] = dict()
        # self.reward_manager.log_info(self, env_ids, self.extras["episode"])
        # self.termination_manager.log_info(self, env_ids, self.extras["episode"])

        self.total_distance[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        

#-------- 4. Get/Set functions--------
    def get_observations(self):
        return self.obs_manager.get_obs_from_group("policy")
    def get_privileged_observations(self):
        return self.obs_manager.get_obs_from_group("privileged")
if __name__ == "__main__":
    env = LocalNavEnv(LocalNavEnvCfg(), LocomotionEnv)
    while True:
        actions = torch.rand(env.num_envs, env.num_actions)
        obs, _,rew, done, extras = env.step(actions)
        if done.any():
            env.reset()

        