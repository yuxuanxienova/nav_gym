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
from nav_gym.nav_legged_gym.envs.legged_nav_env import LeggedNavEnv
from nav_gym.nav_legged_gym.envs.legged_nav_env_config import LeggedNavEnvCfg
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
from nav_gym.nav_legged_gym.envs.hl_nav_env_config import HLNavEnvCfg
import os
from nav_gym import NAV_GYM_ROOT_DIR
class HierarchicalEnv:
    def __init__(self, cfg:HLNavEnvCfg, ll_env_cls:LeggedNavEnv) -> None:
        self.cfg = cfg
        #1. Parse the configuration
        cfg.ll_env_cfg.gym.headless = cfg.gym.headless
        cfg.ll_env_cfg.env.num_envs = cfg.env.num_envs
        #1.1 Create the low-level environment
        self.ll_env: LeggedNavEnv = ll_env_cls(cfg.ll_env_cfg)
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
        #1.3 Load the low-level policy
        model_path = os.path.join(NAV_GYM_ROOT_DIR, "resources/model/low_level/" + "model_4200.pt")
        if os.path.exists(model_path):
            ll_policy = torch.jit.load(model_path).to(self.device)
            print("[info]Low level policy loaded successfully")
        else:
            print("[warning]Low level policy not found")
        
        #2. Prepare mdp helper managers
        self._init_buffers()
        self.sensor_manager = SensorManager(self)
        self.reward_manager = RewardManager(self)
        self.obs_manager = ObsManager(self)
        self.termination_manager = TerminationManager(self)
        self.curriculum_manager = CurriculumManager(self)
        #
        self.reset()
        self.obs_dict = self.obs_manager.compute_obs(self)

    def _init_buffers(self):
        self.obs_dict = dict()
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.ll_reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = dict()

    def step(self, actions):
        self._preprocess_actions(actions)
        self.ll_reset_buf.zero_()
        for _ in range(self.cfg.hl_decimation):
            ll_actions = self._get_ll_actions()
            _, _, dones, _ = self.ll_env.step(ll_actions)
            self.ll_reset_buf |= dones
        # sensors need to be updated
        self.sensor_manager.update()
        self._post_ll_step()

        # print(self.ll_env.reward_manager.episode_sums["contact_forces"])
        return self.obs_dict["policy"], self.rew_buf, self.reset_buf, self.extras
    
    def reset(self):
        """Reset all environment instances."""
        # reset environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.obs_dict = self.obs_manager.compute_obs(self)
        self.obs_buf = self.obs_dict["policy"]
        self.extras["observations"] = self.obs_dict
        # return obs
        return self.obs_buf, self.extras
    
    def reset_idx(env_ids):
        pass
    
    def _post_ll_step():
        pass

    def _preprocess_actions(self, actions: torch.Tensor):
        return actions

    def _get_ll_actions(self):
        """Apply actions to simulation buffers in the environment."""
        raise NotImplementedError
    
    def get_observations(self):
        return self.obs_dict["policy"], self.extras

    def _draw_debug_viz(self):
        pass

if __name__ == "__main__":
    env = HierarchicalEnv(HLNavEnvCfg(), LeggedNavEnv)