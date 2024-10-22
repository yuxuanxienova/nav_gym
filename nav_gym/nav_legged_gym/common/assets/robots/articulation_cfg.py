# legged-gym
from typing import List
from nav_gym.nav_legged_gym.utils.config_utils import configclass
from nav_gym.nav_legged_gym.common.assets.asset_cfg import FileAssetCfg


@configclass
class ArticulationCfg(FileAssetCfg):
    """Configuration for articulated object (loaded from a file)."""

    cls_name = "Articulation"

    enable_dof_force_sensors: bool = False
    """Enable/disable joint force sensor.

    Check: isaacgym_lib/docs/programming/forcesensors.html
    """

    soft_dof_limit_factor = 0.95
    """Fraction specifying the range of DOF limits (parsed from the asset) to use."""

    actuators: List = []
    # @configclass
    # class ActuatorsCfg:
    #     pass

    @configclass
    class InitState(FileAssetCfg.InitState):
        # position targets at 0 actions, Dict("joint_name": value) 'join_name' can be a regex expression
        dof_pos = {".*": 0.0}
        dof_vel = {".*": 0.0}

    init_state = InitState()
