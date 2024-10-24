from nav_gym.nav_legged_gym.utils.config_utils import configclass
from nav_gym.nav_legged_gym.common.assets.robots.articulation_cfg import ArticulationCfg
from nav_gym.nav_legged_gym.common.actuators.actuator_cfg import anymal_d_actuator_cfg
import os

@configclass
class LeggedRobotCfg(ArticulationCfg):
    cls_name = "LeggedRobot"
    feet_names = (
        ".*foot"  # name of the feet rigid bodies (from URDF), used to index body state and contact force tensors
    )
    feet_position_offset = [0.0, 0.0, 0.0]


# Ready to use robots


anymal_d_robot_cfg = LeggedRobotCfg(
    asset_name="anymal_d",
    asset_file = "anymal_d/urdf/anymal_d.urdf",
    feet_names=".*FOOT",
    self_collisions=True,
    replace_cylinder_with_capsule=True,
    init_state=LeggedRobotCfg.InitState(
        pos=(0.0, 0.0, 0.7),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
        },
    ),
    actuators=[{"actuator": anymal_d_actuator_cfg, "dof_names": [".*"]}],
    # actuators=[{"actuator": anymal_simple_actuator_cfg, "dof_names": [".*"], "p_gains": {".*": 80.0},
    #     "d_gains": {".*": 2.0},}],
    randomization=LeggedRobotCfg.Randomization(
        randomize_added_mass=True,
        randomize_friction=True,
        friction_range=(0., 1.5),  # friction coefficients are averaged, mu = 0.5*(mu_terrain + mu_foot)
        # friction_range=(0.75, 1.5),
        added_mass_range=(-5.0, 5.0),
    ),
)

if __name__ == "__main__":
    #check asset file path
    import os
    cfg = anymal_d_robot_cfg
    asset_path = os.path.join(cfg.asset_root, cfg.asset_file)
    if not os.path.exists(asset_path):
        raise FileNotFoundError(f"Asset file does not exist: {asset_path}")
    print("Asset file path is correct.")

