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