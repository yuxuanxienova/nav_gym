import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.common.sensors.sensors_cfg import SensorCfgBase, OmniPatternCfg, FootScanPatternCfg
def my_pattern_func(pattern_cfg, device):
    # Example: generate rays in a circular pattern
    num_rays = 16
    angles = torch.linspace(0, 2 * torch.pi, num_rays, device=device)#Dim:[num_rays]
    ray_directions = torch.stack([torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=1)#Dim:[num_rays,3]
    ray_starts = torch.zeros_like(ray_directions)#Dim:[num_rays,3]
    return ray_starts, ray_directions

def omniscan_pattern(pattern_cfg, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The omniscan pattern for ray casting.

    Args:
        pattern_cfg (OmniPatternCfg): The config for the pattern.
        device (str): The device
    Returns:
        ray_starts (torch.Tensor): The starting positions of the rays
        ray_directions (torch.Tensor): The ray directions

    """
    h = torch.arange(
        -pattern_cfg.horizontal_fov / 2, pattern_cfg.horizontal_fov / 2, pattern_cfg.horizontal_res, device=device
    )
    num_vertical_rays = int(pattern_cfg.vertical_fov / pattern_cfg.vertical_res)
    v = torch.arccos(torch.linspace(1, -1, num_vertical_rays, device=device))
    h = torch.deg2rad(h)
    pitch, yaw = torch.meshgrid(v, h, indexing="xy")
    pitch, yaw = pitch.reshape(-1), yaw.reshape(-1)
    x = torch.sin(pitch) * torch.cos(yaw)
    y = torch.sin(pitch) * torch.sin(yaw)
    z = torch.cos(pitch)
    ray_directions = -torch.stack([x, y, z], dim=1)#Dim: (num_rays, 3)
    ray_starts = torch.zeros_like(ray_directions)#Dim: (num_rays, 3)
    # print("ray_directions", ray_directions)
    return ray_starts, ray_directions


def foot_scan_pattern(pattern_cfg: "FootScanPatternCfg", device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The foot scan pattern for ray casting.
    Args:
        pattern_cfg (FootScanPatternCfg): The config for the pattern.
        device (str): The device
    Returns:
        ray_starts (torch.Tensor): The starting positions of the rays
        ray_directions (torch.Tensor): The ray directions
    """
    pattern = []
    for i, r in enumerate(pattern_cfg.radii):
        for j in range(pattern_cfg.num_points[i]):
            angle = 2.0 * np.pi * j / pattern_cfg.num_points[i]
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = 0.0
            pattern.append([x, y, z])
    ray_starts = torch.tensor(pattern, dtype=torch.float).to(device)

    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(pattern_cfg.direction), device=device)
    return ray_starts, ray_directions
def grid_pattern(pattern_cfg: "GridPatternCfg", device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """A regular grid pattern for ray casting.

    Args:
        pattern_cfg (GridPatternCfg): The config for the pattern.
        device (str): The device
    Returns:
        ray_starts (torch.Tensor): The starting positions of the rays
        ray_directions (torch.Tensor): The ray directions

    """
    y = torch.arange(
        start=-pattern_cfg.width / 2, end=pattern_cfg.width / 2 + 1.0e-9, step=pattern_cfg.resolution, device=device
    )
    x = torch.arange(
        start=-pattern_cfg.length / 2, end=pattern_cfg.length / 2 + 1.0e-9, step=pattern_cfg.resolution, device=device
    )
    grid_x, grid_y = torch.meshgrid(x, y)
    num_rays = grid_x.numel()
    ray_starts = torch.zeros(num_rays, 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 1] = grid_y.flatten()

    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(pattern_cfg.direction), device=device)
    return ray_starts, ray_directions


def velodyne_pattern(pattern_cfg: "VelodynePatternCfg", device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The Velodyne Puck pattern for ray casting.

    Args:
        pattern_cfg (RealSensePatternCfg): The config for the pattern.
        device (str): The device
    Returns:
        ray_starts (torch.Tensor): The starting positions of the rays
        ray_directions (torch.Tensor): The ray directions

    """
    horizontal_angles = torch.arange(0.0, 360.0, pattern_cfg.horizontal_resolution, device=device)
    vertical_angles = torch.arange(
        -pattern_cfg.vertical_fov / 2.0 + pattern_cfg.vertical_offset,
        pattern_cfg.vertical_fov / 2.0 + pattern_cfg.vertical_offset + pattern_cfg.vertical_resolution,
        pattern_cfg.vertical_resolution,
        device=device,
    )
    xy_angles, z_angles = torch.meshgrid(
        torch.deg2rad(horizontal_angles), torch.deg2rad(vertical_angles), indexing="xy"
    )
    ray_directions = torch.cat(
        [torch.cos(xy_angles.unsqueeze(2)), torch.sin(xy_angles.unsqueeze(2)), torch.tan(z_angles.unsqueeze(2))], dim=2
    )
    ray_directions = torch.nn.functional.normalize(ray_directions, p=2.0, dim=-1).view(-1, 3)

    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions