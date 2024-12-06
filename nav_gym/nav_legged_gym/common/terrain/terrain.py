import torch
import numpy as np
import trimesh
from collections import defaultdict

from nav_gym.nav_legged_gym.utils import warp_utils
from nav_gym.nav_legged_gym.common.gym_interface import GymInterface

from isaacgym import gymapi


class Terrain:
    def __init__(self,  num_envs: int, gym_iface: GymInterface) -> None:
        self.gym = gym_iface.gym
        self.sim = gym_iface.sim
        self.device = gym_iface.device
        self.num_envs = num_envs
        self.meshes = defaultdict(trimesh.Trimesh)
        self.wp_meshes = {}

    def add_mesh(self, mesh, name="terrain"):
        if self.meshes[name]:
            mesh = trimesh.util.concatenate(self.meshes[name], mesh)
        self.meshes[name] = mesh
        wp_device = "cuda" if "cuda" in self.device else "cpu"
        self.wp_meshes[name] = warp_utils.convert_to_wp_mesh(mesh.vertices, mesh.faces, wp_device)

    def set_terrain_origins(self):
        self.env_origins = self._compute_env_origins()

    def add_to_sim(self, name="terrain"):
        self._add_ground_plane_to_sim()

    def _add_ground_plane_to_sim(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)


    def _compute_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """

        env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = 1.0
        env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        env_origins[:, 2] = 0.0
        return env_origins

