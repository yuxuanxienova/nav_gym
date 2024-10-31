from typing import Tuple, Callable, Optional, Any, List
from nav_gym.nav_legged_gym.common.sensors.sensor_utils import my_pattern_func, omniscan_pattern, foot_scan_pattern, grid_pattern
from nav_gym.nav_legged_gym.utils.config_utils import configclass

#----------------------------Sensor Base-------------------------------
@configclass
class SensorCfgBase:
    enable_debug_vis: bool = False

#----------------------------Raycaster Pattern--------------------------------
@configclass
class OmniPatternCfg(SensorCfgBase):
    # width: int = 128
    # height: int = 72
    # far_plane: float = 4.0
    horizontal_fov: float = 180
    horizontal_res: float = 3.0#8.0
    vertical_fov: float = 180
    vertical_res: float = 3.0#8.0
    pattern_func: Callable = omniscan_pattern  # realsense_pattern
@configclass
class GridPatternCfg(SensorCfgBase):
    resolution: float = 0.1
    width: float = 1.0
    length: float = 1.6
    max_xy_drift: float = 0.05
    direction: Tuple = (0.0, 0.0, -1.0)
    pattern_func: Callable = my_pattern_func    

@configclass
class FootScanPatternCfg(SensorCfgBase):
    radii: Tuple[float, ...] = (0.08, 0.16, 0.26, 0.36, 0.48)
    num_points: Tuple[int, ...] = (6, 8, 10, 12, 16)
    direction: Tuple = (0.0, 0.0, -1.0)
    pattern_func: Callable = foot_scan_pattern

@configclass
class GridPatternCfg:
    resolution: float = 0.1
    width: float = 1.0
    length: float = 1.6
    direction: Tuple = (0.0, 0.0, -1.0)
    pattern_func: Callable = grid_pattern
#----------------------------Raycaster -------------------------------
@configclass
class RaycasterCfg(SensorCfgBase):
    class_name: str = "Raycaster"
    terrain_mesh_names: Tuple[str, ...] = ("terrain",)
    robot_name: str = "robot"
    body_attachement_name: str = "base"
    attachement_pos: Tuple = (0.0, 0.0, 0.4)
    attachement_quat: Tuple = (0.0, 0.0, 0.0, 1.0)
    attach_yaw_only: bool = True  # do not use the roll and pitch of the robot to update the rays
    default_hit_value: float = -10.0  # which value to return when a ray misses the hit
    default_hit_distance: float = 10.0  # which distance to return when a ray misses the hit
    pattern_cfg: Any = GridPatternCfg()
    post_process_func: Optional[Callable] = None  # function to apply to the raycasted values
    visualize_friction: bool = False
    max_xy_drift = 0.00
    max_z_drift = 0.00

@configclass
class OmniScanRaycasterCfg(RaycasterCfg):
    class_name: str = "Raycaster"
    terrain_mesh_names: Tuple[str, ...] = ("terrain",)
    robot_name: str = "robot"
    body_attachement_name: str = "base"
    attachement_pos: Tuple = (0.0, 0.0, 0.0)
    attachement_quat: Tuple = (0.0, 0.0, 0.0, 1.0)
    attach_yaw_only: bool = True
    default_hit_value: float = -10.0  # which value to return when a ray misses the hit
    default_hit_distance: float = 10.0  # which distance to return when a ray misses the hit
    pattern_cfg: Any = OmniPatternCfg()
    post_process_func: Optional[Callable] = None  # function to apply to the raycasted values
    visualize_friction: bool = False
    max_xy_drift = 0.00
    max_z_drift = 0.00
    #new parameter
    max_distance: float = 10.0

@configclass
class FootScanCfg(RaycasterCfg):
    class_name: str = "Raycaster_footscan"
    yaw_attachment_name: str = "base"  # This version follows the yaw of the base
    attach_yaw_only: bool = True
    pattern_cfg = FootScanPatternCfg()
    attachement_pos: Tuple = (0.0, 0.0, 20.0)
    debug_color: Tuple = (0.0, 1.0, 0.0)

    def normalize(self, *args, **kwargs):
        print(args, kwargs)








