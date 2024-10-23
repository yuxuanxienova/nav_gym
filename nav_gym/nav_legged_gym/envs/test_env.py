# isaac-gym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_apply

# python
from copy import deepcopy
import torch
import numpy as np

# legged-gym
from nav_gym.nav_legged_gym.envs.base_env import BaseEnv
from nav_gym.nav_legged_gym.envs.legged_env import LeggedEnv
from nav_gym.nav_legged_gym.envs.legged_env_config import LeggedEnvCfg
from nav_gym.nav_legged_gym.common.assets.robots.legged_robots.legged_robot import LeggedRobot
from nav_gym.nav_legged_gym.common.sensors.sensors import SensorBase, Raycaster
from nav_gym.nav_legged_gym.utils.math_utils import wrap_to_pi
from nav_gym.nav_legged_gym.common.terrain.terrain_unity import TerrainUnity
from nav_gym.nav_legged_gym.utils.visualization_utils import BatchWireframeSphereGeometry

class TestEnv(LeggedEnv):
    robot: LeggedRobot
    cfg: LeggedEnvCfg
    def __init__(self, cfg: LeggedEnvCfg):
        super().__init__(cfg)
        self.raycaster = Raycaster(cfg=cfg.raycaster,env=self)
        #----------------------4. Prepare Debug Usage------------------------
        self.sphere_geoms_red = BatchWireframeSphereGeometry(num_spheres=1,radius=0.1, num_lats=4, num_lons=4, pose=None, color=(1, 0, 0))
        self.sphere_geoms_green = BatchWireframeSphereGeometry(num_spheres=1,radius=0.1, num_lats=4, num_lons=4, pose=None, color=(0, 1, 0))
        self.sphere_geoms_blue = BatchWireframeSphereGeometry(num_spheres=1,radius=0.1, num_lats=4, num_lons=4, pose=None, color=(0, 0, 1))

    def _post_physics_step(self):
        self.raycaster.update(self.dt)
        super()._post_physics_step()
    def render(self, sync_frame_time=True):
        super().render(sync_frame_time)
        self.raycaster.debug_vis(self) 
        #Drawing the Axis
        sphere_pos_init = torch.tensor([0,0,0]).unsqueeze(0)
        self.sphere_geoms_red.draw(sphere_pos_init , self.gym, self.viewer, self.envs[0])
        x_offset = torch.tensor([0.5,0,0],device=self.device)
        y_offset = torch.tensor([0,0.5,0],device=self.device)
        z_offset = torch.tensor([0,0,0.5],device=self.device)
        self.sphere_geoms_red.draw(sphere_pos_init + x_offset, self.gym, self.viewer, self.envs[0])
        self.sphere_geoms_green.draw(sphere_pos_init + y_offset, self.gym, self.viewer, self.envs[0])
        self.sphere_geoms_blue.draw(sphere_pos_init + z_offset, self.gym, self.viewer, self.envs[0])
        
