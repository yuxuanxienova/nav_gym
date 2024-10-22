from dataclasses import MISSING
import torch
import warp as wp
import numpy as np
import open3d as o3d

from isaacgym.torch_utils import quat_apply
from nav_gym.nav_legged_gym.utils.math_utils import matrix_from_quat, quat_apply_yaw, torch_rand_float, yaw_quat, quat_rotate_inverse
from nav_gym.nav_legged_gym.utils.visualization_utils import BatchWireframeSphereGeometry
from nav_gym.nav_legged_gym.utils.warp_utils import ray_cast
from nav_gym.nav_legged_gym.common.sensors.sensors_cfg import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs import BaseEnv


class SensorBase:
    is_up_to_date: bool = MISSING
    def __init__(self, cfg, env):
        # prepare some buffers
        # enable corresponding sensors in sim
        raise NotImplementedError()

    def spawn(self, env_id):
        pass

    def update(self, dt: float, env_ids=None):
        # compute stuff
        raise NotImplementedError()

    def get_data(self):
        # returns sensor data
        raise NotImplementedError()

    def reset(self):
        pass

    def needs_update(self):
        self.is_up_to_date = False

class Raycaster(SensorBase):
    def __init__(self, cfg: RaycasterCfg, env: "BaseEnv"):
        #1. Configuration and Environment Setup
        self.cfg = cfg
        self.terrain_mesh = env.terrain.wp_meshes
        self.robot = getattr(env, cfg.robot_name)
        self.body_idx, _ = self.robot.find_bodies(cfg.body_attachement_name)
        self.num_envs = self.robot.num_envs
        self.device = self.robot.device

        #2. Raycasting Setup
        self.ray_starts, self.ray_directions = cfg.pattern_cfg.pattern_func(cfg.pattern_cfg, self.device)

        #3. Sensor Drift
        self.num_rays = len(self.ray_directions)
        self.drift = torch.zeros(env.num_envs, self.num_rays, 3, device=self.device)
        self.drift[..., :2] = torch_rand_float(
            -cfg.max_xy_drift, cfg.max_xy_drift, (env.num_envs, 2), device=self.device
        ).unsqueeze(1)
        self.drift[..., 2] = torch_rand_float(-cfg.max_z_drift, cfg.max_z_drift, (env.num_envs, 1), device=self.device)
        # self.drift[..., 0] = +0.2
        # self.drift[..., 1] = -0.15
        # self.drift[..., 2] = -0.

        #3. Sensor Attachment Transformation
        offset_pos = torch.tensor(list(cfg.attachement_pos), device=self.device)
        offset_quat = torch.tensor(list(cfg.attachement_quat), device=env.device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos

        #4. Environment Replication
        self.ray_starts = self.ray_starts.repeat(self.num_envs, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)
        self.ray_starts[:, :, :2] += self.drift[..., :2]

        #5. Buffers Initialization
        self.ray_hits_world = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)
        self.ray_distances = torch.zeros(self.num_envs, self.num_rays, 1, device=self.device)
        self.sphere_geom = None
        self.is_up_to_date = False

    def update(self, dt, env_ids=...):
        """Perform raycasting on the terrain.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
        """
        # states = self.robot.rigid_body_states[env_ids, self.body_idx, :].squeeze(1)
        # pos = states[..., :3]
        # quats = states[..., 3:7]
        env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is ... else env_ids
        #1. Robot State Retrieval
        pos = self.robot.root_pos_w
        quats = self.robot.root_quat_w

        #2. Transforming Rays to World Frame
        if self.cfg.attach_yaw_only:
            ray_starts_world = quat_apply_yaw(quats.repeat(1, self.num_rays), self.ray_starts[env_ids]) + pos.unsqueeze(1)
            ray_directions_world = self.ray_directions[env_ids]
        else:
            ray_starts_world = quat_apply(quats.repeat(1, self.num_rays), self.ray_starts[env_ids]) + pos.unsqueeze(1)
            ray_directions_world = quat_apply(quats.repeat(1, self.num_rays), self.ray_directions[env_ids])
        #2.(Optional)----Debug Raycasting----
        # self.num_rays=3
        # self.drift = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)
        # self.drift[..., :2] = torch_rand_float(
        #     -self.cfg.max_xy_drift, self.cfg.max_xy_drift, (self.num_envs, 2), device=self.device
        # ).unsqueeze(1)
        # self.drift[..., 2] = torch_rand_float(-self.cfg.max_z_drift, self.cfg.max_z_drift, (self.num_envs, 1), device=self.device)
        # self.ray_hits_world = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)
        # self.ray_distances = torch.zeros(self.num_envs, self.num_rays, 1, device=self.device)
        # ray_starts_debug = torch.tensor([[0,0,0] , [0,0,0] , [0,0,0]]).repeat(len(env_ids),1).to(self.device).float()#Dim:[num_envs,3]
        # ray_starts_world = quat_apply_yaw(quats.repeat(1, 1), ray_starts_debug ) + pos.unsqueeze(1)

        # ray_directions_debug = torch.tensor([[0,0,-1], [0,1,0] , [1,0,0]]).repeat(len(env_ids),1).to(self.device).float()#Dim:[num_envs,3]
        # ray_directions_world = quat_apply_yaw(quats.repeat(1, 1), ray_directions_debug) 
        #-------------------------------

        #3. Raycasting
        #ray_starts_world: torch.Size([num_envs, num_rays, 3])
        #ray_directions_world: torch.Size([num_envs, num_rays, 3])
        self.ray_hits_world[env_ids], self.ray_distances[env_ids] = ray_cast(ray_starts_world, ray_directions_world, self.terrain_mesh)
        self.ray_hits_world[env_ids, :, 2] += self.drift[env_ids, :, 2]
        self.is_up_to_date = True

    def get_data(self):
        """Returns the ray hit positions, ensuring that any NaN values 
        (from rays that didn't hit anything) are replaced with a default value."""
        if not self.is_up_to_date:
            self.update(0.)
        return torch.nan_to_num(self.ray_hits_world, posinf=self.cfg.default_hit_value)
    
    def get_distances(self) -> torch.Tensor:
        """Returns the distances to the ray hit points, ensuring that any NaN values 
        (from rays that didn't hit anything) are replaced with a default value."""
        return torch.nan_to_num(self.ray_distances, posinf=self.cfg.default_hit_distance)
    
    def debug_vis(self, env: "BaseEnv"):
        """Visualizes the ray hits in the simulation environment"""
        #1. Initializing Visualization Geometry
        if self.sphere_geom is None:self.sphere_geom = BatchWireframeSphereGeometry(self.num_envs * self.num_rays, 0.02, 4, 4, None, color=(0, 1, 0))

        #2. Drawing Ray Hits
        points = self.ray_hits_world.clone()
        points[..., :2] -= self.drift[..., :2]
        self.sphere_geom.draw(points, env.gym, env.viewer, env.envs[0])







