# legged-gym
from typing import Dict, List, Tuple
from nav_gym.nav_legged_gym.utils.config_utils import configclass
from nav_gym.nav_legged_gym.envs.base_env_config import BaseEnvCfg, EnvCfg, ControlCfg
from nav_gym.nav_legged_gym.common.assets.robots.legged_robots.legged_robots_cfg import LeggedRobotCfg,anymal_d_robot_cfg
from nav_gym.nav_legged_gym.common.gym_interface.gym_interface_cfg import GymInterfaceCfg, ViewerCfg
from nav_gym.nav_legged_gym.common.sensors.sensors_cfg import SensorsCfg,RaycasterCfg
import nav_gym.nav_legged_gym.common.observations.observations as O
import nav_gym.nav_legged_gym.common.rewards.rewards as R
import nav_gym.nav_legged_gym.common.terminations.terminations as T
import nav_gym.nav_legged_gym.common.curriculum.curriculum as C



@configclass
class CommandsCfg:
    resampling_time = 10.0  # time before commands are changed [s]
    heading_command = True  # if true: compute ang vel command from heading error
    rel_standing_envs = 0.02  # percentage of the robots are standing
    rel_heading_envs = 1.0  # percentage of the robots follow heading command (the others follow angular velocity)

    @configclass
    class Ranges:
        lin_vel_x: List = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y: List = [-1.0, 1.0]  # min max [m/s]
        ang_vel_yaw: List = [-1.5, 1.5]  # min max [rad/s]
        heading: List = [-3.14, 3.14]  # [rad]

    ranges = Ranges()


@configclass
class RandomizationCfg:
    # randomize_friction: bool = True
    # friction_range: Tuple = (0.5, 1.25)
    # randomize_base_mass: bool = False
    # added_mass_range: Tuple = (-1.0, 1.0)
    push_robots: bool = True
    push_interval_s: float = 15  # push applied each time interval [s]
    init_pos: Tuple = (-1.0, 1.0)  # max xy position added to default position [m]
    init_yaw: Tuple = (-3.14, 3.14)  # max yaw angle added to default orientation [rad]
    init_roll_pitch: Tuple = (0.0, 0.0)  # max roll and pitch angles added to default orientation [rad]
    push_vel: Tuple = (-1.0, 1.0)  # velocity offset added by push [m/s]
    external_force: Tuple = (-0.0, 0.0)  # wind force applied at base, constant over episode [N]
    external_torque: Tuple = (-0.0, 0.0)  # wind torque applied at base, constant over episode [Nm]
    external_foot_force: Tuple = (-0.0, 0.0)  # wind force applied at feet, constant over episode [N]


@configclass
class ObservationsCfg:
    @configclass
    class Policy:
        # optinal parameters: scale, clip([min, max]), noise
        add_noise: bool = True  # turns off the noise in all observations
        base_lin_vel: dict = {"func": O.base_lin_vel, "noise": 0.1}
        base_ang_vel: dict = {"func": O.base_ang_vel, "noise": 0.2}
        projected_gravity: dict = {"func": O.projected_gravity, "noise": 0.05}
        velocity_commands: dict = {"func": O.velocity_commands}
        dof_pos: dict = {"func": O.dof_pos, "noise": 0.01}
        dof_vel: dict = {"func": O.dof_vel, "noise": 1.5}
        actions: dict = {"func": O.actions}
        # height_scan: dict = {"func": O.ray_cast, "noise": 0.1, "sensor": "height_scanner", "clip": (-1, 1.0)}
        # bpearl: dict = {"func_name": O.ray_cast, "noise": 0.1, "sensor": "bpearl_front"}
        # bpearl2: dict = {"func_name": O.ray_cast, "noise": 0.1, "sensor": "bpearl_rear"}

    policy = Policy()


@configclass
class RewardsCfg:
    # general params
    only_positive_rewards: bool = False
    # reward functions
    termination = {"func": R.termination, "scale": -0.0}
    tracking_lin_vel = {"func": R.tracking_lin_vel, "scale": 1.0, "std": 0.25}
    tracking_ang_vel = {"func": R.tracking_ang_vel, "scale": 0.5, "std": 0.25}
    lin_vel_z = {"func": R.lin_vel_z, "scale": -2.0}
    ang_vel_xy = {"func": R.ang_vel_xy, "scale": -0.05}
    torques = {"func": R.torques, "scale": -0.00002}
    dof_acc = {"func": R.dof_acc, "scale": -2.5e-7}
    feet_air_time = {"func": R.feet_air_time, "scale": 0.5, "time_threshold": 0.5}
    collision = {"func": R.collision, "scale": -0.25, "bodies": ".*(THIGH|SHANK)"}
    action_rate = {"func": R.action_rate, "scale": -0.01}
    dof_vel = {"func": R.dof_vel, "scale": -0.0}
    stand_still = {"func": R.stand_still, "scale": -0.0}
    base_height = {"func": R.base_height, "scale": -0.0, "height_target": 0.5, "sensor": "ray_caster"}
    flat_orientation = {"func": R.flat_orientation, "scale": -0.0}
    # stumble = {"func": "stumble", "scale": -1.0, "hv_ratio": 2.0}
    # contact_forces = {"func": "contact_forces", "scale": -0.01, "max_contact_force": 450}


@configclass
class TerminationsCfg:
    # general params
    reset_on_termination: bool = True
    time_out = {"func": T.time_out}
    illegal_contact = {"func": T.illegal_contact, "bodies": "base"}
    bad_orientation = None
    dof_torque_limit = None
    dof_pos_limit = None


@configclass
class CurriculumCfg:
    # general params
    # terrain_levels = {"func": C.terrain_levels_vel, "mode": "on_reset"}
    max_lin_vel_command = None


@configclass
class LeggedEnvCfg(BaseEnvCfg):

    # common configuration (from base env)
    env = EnvCfg(num_envs=1, num_actions=12, send_timeouts=True, episode_length_s=20)
    gym = GymInterfaceCfg(viewer=ViewerCfg(eye=(10, 0, 6), target=(11, 5, 3)))
    control = ControlCfg(decimation=4, action_scale=0.5, action_clipping=100.0)

    # legged-env specific configurations
    # -- scene designing
    robot = anymal_d_robot_cfg
    sensors = SensorsCfg()
    raycaster = RaycasterCfg()
    # -- mdp signals
    randomization = RandomizationCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()
    commands = CommandsCfg()
