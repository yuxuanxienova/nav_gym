
from nav_gym.nav_legged_gym.common.assets.robots.legged_robots.legged_robots_cfg import LeggedRobotCfg,anymal_d_robot_cfg
from nav_gym.nav_legged_gym.common.gym_interface.gym_interface_cfg import GymInterfaceCfg, ViewerCfg,SimParamsCfg,PhysxCfg
from nav_gym.nav_legged_gym.common.sensors.sensors_cfg import RaycasterCfg,OmniScanRaycasterCfg,FootScanCfg,GridPatternCfg
import nav_gym.nav_legged_gym.common.observations.observations as O
import nav_gym.nav_legged_gym.common.rewards.rewards as R
import nav_gym.nav_legged_gym.common.terminations.terminations as T
import nav_gym.nav_legged_gym.common.curriculum.curriculum as C
from typing import Dict, List, Tuple
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.nav_legged_gym.common.commands.commands_cfg import UnifromVelocityCommandCfg
from nav_gym.nav_legged_gym.envs.config_locomotion_env import LocomotionEnvCfg
from nav_gym.nav_legged_gym.utils.config_utils import configclass
from nav_gym.nav_legged_gym.envs.modules.utils import distance_compare
#---Local Navigation Module Config Parameters---
DIS_THRE = 2.0
FOV_RANGE = 2.0
GOAL_RADIUS = 2.0

NUM_HISTORY = 50
NUM_NODES = 20
SENSOR_HEIGHT = 5.0  # NOTE: be careful with multi-floor env.
#-------------------------------------------------
class LocalNavEnvCfg:
    ll_env_cfg = LocomotionEnvCfg()
    hl_decimation: int = 24 #high level control loop: interval = hl_decimation * ll_env_cfg.dt (4 * 0.02 = 0.08[s])
    max_x_vel = 1.0
    max_y_vel = 0.5
    max_yaw_rate = 1.25

    # for beta distribution
    vel_cmd_max: Tuple[float, float, float] = (max_x_vel, max_y_vel, max_yaw_rate)  # x, y, yaw
    vel_cmd_scale: Tuple[float, float, float] = (2.0 * max_x_vel, 2.0 * max_y_vel, 2.0 * max_yaw_rate)  # x, y, yaw
    vel_cmd_offset: Tuple[float, float, float] = (-max_x_vel, -max_y_vel, -max_yaw_rate)

    class env:
        """Common configuration for environment."""

        num_envs: int = 3
        """Number of environment instances."""

        num_actions: int = 3  
        """The size of action space for the defined environment MDP."""

        episode_length_s: float = 10.0
        """Episode length in seconds."""

        send_timeouts: bool = True  # send time out information to the algorithm
        """Whether to send episode time-out information (added as part of infos)."""

        enable_debug_vis: bool = True


    gym = GymInterfaceCfg()

    class control:
        """Control configuration for stepping the environment."""

        decimation: int = 4 #control loop: interval = control.decimation * sim_params.dt (4 * 0.005 = 0.02[s])
        """Number of times to apply control action, i.e. number of simulation time-steps per policy time-step."""

        action_clipping: float = 100.0
        """Clipping of actions provided to the environment."""

        action_scale: float = 0.5
        """Scaling of input actions provided to the environment."""


    robot = anymal_d_robot_cfg

    class sensors:
        raycasters_dict = {
                        #  "omni_scanner1": OmniScanRaycasterCfg(),
                        # "height_scanner": RaycasterCfg(attachement_pos=(0.0, 0.0, 20.0), attach_yaw_only=True, pattern_cfg=GridPatternCfg(width=1.0, length=2.0),max_xy_drift=0.075,max_z_drift=0.075),
                        "height_scanner" : RaycasterCfg(
                            attachement_pos=(0.0, 0.0, SENSOR_HEIGHT),
                            attach_yaw_only=True,
                            pattern_cfg=GridPatternCfg(resolution=0.25, width=FOV_RANGE * 2, length=FOV_RANGE * 2),
                        ),
                          }
    class randomization:
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
        #---local navigation---
        keep_memory_prob: float = 0.3

    class observations:
        class prop:
            # add this to every group
            add_noise: bool = True  # turns off the noise in all observations

            llc_prop: dict = {"func": O.llc_obs, "name": "policy"}  
        class ext:
            # add this to every group
            add_noise: bool = True
            height_scan: dict = {"func": O.ray_cast, "noise": 0.1, "sensor": "height_scanner", "clip": (-1.0, 1.0)}
            # height_scan: dict = {
            # "func": O.height_trav_map,
            # "occlusion_fill_height": 0.0,
            # "occlusion_fill_travers": -1.0,  # 0.0: not traversable, 1.0: traversable
            # # "func": OE.ray_cast_exp,
            # "noise": 0.05,
            # "sensor": "height_scanner",
            # "max_height": 0.5,  # from base. Occluded if higher
            # "min_height": -1.5,  #  from base. Occluded if lower
            # }
        class history:
            add_noise: bool = True
            wp_pos_history: dict = {
                "func": O.wp_pos_history,
                "clip_range": 10.0,
                "num_points": 5,
                "decimation": 5,
                "noise": 0.1,
            }
        class memory:
            add_noise: bool = True
            graph_poses: dict = {
                "func": O.node_positions_times,
                "num_points": NUM_NODES,
                "counter_limit": 10,
                "pose_noise": 0.1,
            }
    class rewards:
        # general params
        only_positive_rewards: bool = False
        # reward functions
        # goal_position = {"func": R.tracking_dense, "max_error": GOAL_RADIUS, "scale": 0.5}
        # goal_dot = {"func": R.goal_dot_prod_decay, "goal_radius": GOAL_RADIUS, "max_magnitude": 0.5, "scale": 0.2}
        goal_tracking_dense_dot = {"func": R.goal_tracking_dense_dot, "goal_radius": GOAL_RADIUS, "max_magnitude": 1, "scale": 10}

        dof_vel_legs = {"func": R.dof_vel_selected, "scale": -1.0e-6, "dofs": ".*(HAA|HFE|KFE)"}
        dof_acc_legs = {"func": R.dof_acc_selected, "scale": -1.0e-8, "dofs": ".*(HAA|HFE|KFE)"}
        torque_limits = {"func": R.torque_limits, "scale": -1.0e-6, "soft_ratio": 0.95}

        action_limits = {"func": R.action_limits_penalty, "scale": -0.1, "soft_ratio": 0.95}
        # near_goal_stability: dict = {"func": R.near_goal_stability, "std": 1.0, "threshold": 1.0, "scale": 0.1}

        # Exploration (when explicit memory is used)
        global_exp_volume: dict = {"func": R.global_exp_volume, "scale": 0.05}
        exp_bonus: dict = {"func": R.exp_bonus, "max_count": 10.0, "scale": 0.001}
        face_front = {
            "func": R.face_front,
            "angle_limit": 0.78,
            "min_vel": 0.2,
            "scale": 0.025,
        }  # To account for the camera FOV. Vel direction in baseframe < 45 degrees

        action_rate = {"func": R.action_rate, "scale": -0.01}
        action_rate2 = {"func": R.action_rate_2, "scale": -0.01}

    class terminations:
        # general params
        reset_on_termination: bool = True
        time_out = {"func": T.time_out}
        illegal_contact ={"func": T.illegal_contact, "bodies": "base"}
        bad_orientation = None
        dof_torque_limit = None
        dof_pos_limit = None

    class curriculum:
        # general params
        # terrain_levels = {"func": C.terrain_levels_vel, "mode": "on_reset"}
        max_lin_vel_command = None

    # class commands:
    #     resampling_time = 10.0  # time before commands are changed [s]
    #     heading_command = True  # if true: compute ang vel command from heading error
    #     rel_standing_envs = 0.02  # percentage of the robots are standing
    #     rel_heading_envs = 1.0  # percentage of the robots follow heading command (the others follow angular velocity)
    #     class ranges:
    #         lin_vel_x: List = [-1.0, 1.0]  # min max [m/s]
    #         lin_vel_y: List = [-1.0, 1.0]  # min max [m/s]
    #         ang_vel_yaw: List = [-1.5, 1.5]  # min max [rad/s]
    #         heading: List = [-3.14, 3.14]  # [rad]
    commands = UnifromVelocityCommandCfg()
#-----------------------------Local Navigation Module Config--------------------------------


    class memory:
        use_local_map = False
        max_node_cnt = NUM_NODES + 1
        history_len = NUM_HISTORY + 1
        wp_history_len = NUM_HISTORY + 1
        class comparator:
            func = distance_compare
            params = {"thre": DIS_THRE}


if __name__ == "__main__":
    cfg = LocalNavEnvCfg()

