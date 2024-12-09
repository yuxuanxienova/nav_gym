
from nav_gym.nav_legged_gym.common.assets.robots.legged_robots.legged_robots_cfg import LeggedRobotCfg,anymal_d_robot_cfg,anymal_d_robot_pae_cfg
from nav_gym.nav_legged_gym.common.gym_interface.gym_interface_cfg import GymInterfaceCfg, ViewerCfg,SimParamsCfg,PhysxCfg
from nav_gym.nav_legged_gym.common.sensors.sensors_cfg import RaycasterCfg,OmniScanRaycasterCfg,FootScanCfg,GridPatternCfg,BaseScanCfg
import nav_gym.nav_legged_gym.common.observations.observations as O
import nav_gym.nav_legged_gym.common.rewards.rewards as R
import nav_gym.nav_legged_gym.common.terminations.terminations as T
import nav_gym.nav_legged_gym.common.curriculum.curriculum as C
from typing import Dict, List, Tuple
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.nav_legged_gym.common.commands.commands_cfg import UnifromVelocityCommandCfg
from nav_gym import NAV_GYM_ROOT_DIR
class LocomotionPAEEnvCfg:
    class env:
        """Common configuration for environment."""

        num_envs: int = 4096
        """Number of environment instances."""

        num_actions: int = 12  # joint positions, velocities or torques
        """The size of action space for the defined environment MDP."""

        episode_length_s: float = 100.0
        """Episode length in seconds."""

        send_timeouts: bool = True  # send time out information to the algorithm
        """Whether to send episode time-out information (added as part of infos)."""

        enable_debug_vis: bool = False


    gym = GymInterfaceCfg()
    gym.sim_params.dt = 0.0025

    class control:
        """Control configuration for stepping the environment."""

        decimation: int = 8 #control loop: interval = control.decimation * sim_params.dt (4 * 0.005 = 0.02[s])
        """Number of times to apply control action, i.e. number of simulation time-steps per policy time-step."""

        action_clipping: float = 20.0
        """Clipping of actions provided to the environment."""

        action_scale: float = 0.2
        """Scaling of input actions provided to the environment."""


    robot = anymal_d_robot_pae_cfg


    class sensors:
        raycasters_dict = {
                        # "height_scanner": RaycasterCfg(attachement_pos=(0.0, 0.0, 20.0), attach_yaw_only=True, pattern_cfg=GridPatternCfg(width=2.0, length=3.0, resolution=0.2),max_xy_drift=0.075,max_z_drift=0.075),
                        #  "base_scan_center" : BaseScanCfg(body_attachement_name="base", enable_debug_vis=False)  # to check base height
                          }
    class randomization:
        # randomize_friction: bool = True
        friction_range: Tuple = (0.5, 1.25)
        randomize_base_mass: bool = True
        added_mass_range: Tuple = (-1.0, 1.0)
        push_robots: bool = True
        push_interval_s: float = 15  # push applied each time interval [s]
        init_pos: Tuple = (-1.0, 1.0)  # max xy position added to default position [m]
        init_yaw: Tuple = (-3.14, 3.14)  # max yaw angle added to default orientation [rad]
        init_roll_pitch: Tuple = (0.0, 0.0)  # max roll and pitch angles added to default orientation [rad]
        push_vel: Tuple = (-1.0, 1.0)  # velocity offset added by push [m/s]
        external_force: Tuple = (-0.0, 0.0)  # wind force applied at base, constant over episode [N]
        external_torque: Tuple = (-0.0, 0.0)  # wind torque applied at base, constant over episode [Nm]
        external_foot_force: Tuple = (-0.0, 0.0)  # wind force applied at feet, constant over episode [N]


    class observations:

        class prop:
            # --add this to every group--
            add_noise: bool = False # turns off the noise in all observations
            #---------------------------
            # velocity_commands: dict = {"func": O.velocity_commands}
            dof_pos: dict = {"func": O.dof_pos, "noise": 0.0, "scale":1.0}
            # dof_prev_pos: dict = {"func": O.dof_pos_history_selected,"noise": 0.01,"dofs": ".*(HAA|HFE|KFE)","hist_index": -4,}  # 0.005 x 4
            dof_vel: dict = {"func": O.dof_vel, "noise": 0.0,"scale": 0.05}
            # dof_prev_vel: dict = {"func": O.dof_vel_history_selected, "noise": 3e-1, "dofs": ".*", "hist_index": -4}
            actions: dict = {"func": O.actions, "scale":1.0}

        # class exte:
        #     # --add this to every group--
        #     add_noise: bool = True  # turns off the noise in all observations
        #     #--------------------------- 
        #     height_scan: dict = {"func": O.ray_cast, "noise": 0.1, "sensor": "height_scanner", "clip": (-1, 1.0)}
        class priv:
            # --add this to every group--
            add_noise: bool = False # turns off the noise in all observations
            #---------------------------
            base_lin_vel: dict = {"func": O.base_lin_vel, "noise": 0.0, "scale": 2.0}
            base_ang_vel: dict = {"func": O.base_ang_vel, "noise": 0.0, "scale": 0.25}
            projected_gravity: dict = {"func": O.projected_gravity, "noise": 0.0, "scale": 1.0}
            
            

        class fld:
            # --add this to every group--
            add_noise: bool = False
            #---------------------------
            fld_latent_phase_sin: dict = {"func": O.fld_latent_phase_sin, "noise": 0.0}
            fld_latent_phase_cos: dict = {"func": O.fld_latent_phase_cos, "noise": 0.0}
            # fld_latent_others: dict = {"func": O.fld_latent_others, "noise": 0.0}
            fld_latent_onehot: dict = {"func": O.fld_one_hot, "noise": 0.0}
 


    class rewards:
        # general params
        only_positive_rewards: bool = False
        # reward functions
        # termination = {"func": R.termination, "scale": -1}
        # tracking_lin_vel = {"func": R.tracking_lin_vel, "scale": 2.0, "std": 0.25}
        # tracking_ang_vel = {"func": R.tracking_ang_vel, "scale": 1.0, "std": 0.25}
        # base_motion = {"func": R.base_motion, "scale": 0.5, "std_z": 0.5, "std_angvel": 2.0}
        # base_height = {"func": R.base_height, "scale": -0.0, "height_target": 0.5, "sensor": "ray_caster"}
        # feet_air_time = {"func": R.feet_air_time, "scale": 0.4, "time_threshold": 0.5}
        feet_acc = {"func": R.feet_acc, "scale": -0.0001}
        action_rate = {"func": R.action_rate, "scale": -0.005}
        collision_THIGHSHANK = {"func": R.collision, "scale": -1.0, "bodies": ".*(THIGH|SHANK)"}
        collision_base = {"func": R.collision, "scale": -1.0, "bodies": "base"}
        dof_acc = {"func": R.dof_acc, "scale": -5e-7}
        torques = {"func": R.torques, "scale": -0.00001}
        torque_limits = {"func": R.torque_limits, "scale": -0.01, "soft_ratio": 0.9}
        # dof_vel = {"func": R.dof_vel, "scale": -0.0}
        # survival = {"func": R.survival, "scale": 1.0}
        # contact_forces = {"func": "contact_forces", "scale": -0.01, "max_contact_force": 450}
        #-----fld rewards-----
        reward_tracking_reconstructed_lin_vel = {"func": R.reward_tracking_reconstructed_lin_vel, "scale": 1.0, "exp_scale": 1.0}
        reward_tracking_reconstructed_ang_vel = {"func": R.reward_tracking_reconstructed_ang_vel, "scale": 1.0, "exp_scale": 0.8}
        reward_tracking_reconstructed_projected_gravity = {"func": R.reward_tracking_reconstructed_projected_gravity, "scale": 1.0, "exp_scale": 3.0}
        reward_tracking_reconstructed_dof_pos_leg_fl = {"func": R.reward_tracking_reconstructed_dof_pos_leg_fl, "scale": 1.0, "exp_scale": 2.0}
        reward_tracking_reconstructed_dof_pos_leg_hl = {"func": R.reward_tracking_reconstructed_dof_pos_leg_hl, "scale": 1.0, "exp_scale": 2.0}
        reward_tracking_reconstructed_dof_pos_leg_fr = {"func": R.reward_tracking_reconstructed_dof_pos_leg_fr, "scale": 1.0, "exp_scale": 2.0}
        reward_tracking_reconstructed_dof_pos_leg_hr = {"func": R.reward_tracking_reconstructed_dof_pos_leg_hr, "scale": 1.0, "exp_scale": 2.0}
        # reward_tracking_reconstructed_feet_pos_fl = {"func": R.reward_tracking_reconstructed_feet_pos_fl, "scale": 1.0, "std": 0.25}
        # reward_tracking_reconstructed_feet_pos_fr = {"func": R.reward_tracking_reconstructed_feet_pos_fr, "scale": 1.0, "std": 0.25}
        # reward_tracking_reconstructed_feet_pos_hl = {"func": R.reward_tracking_reconstructed_feet_pos_hl, "scale": 1.0, "std": 0.25}
        # reward_tracking_reconstructed_feet_pos_hr = {"func": R.reward_tracking_reconstructed_feet_pos_hr, "scale": 1.0, "std": 0.25}


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

    class terrain_unity:
        terrain_file:str = "/terrain/Plane1.obj"
        translation: Tuple = (0.0, 0.0, 0.0)

        env_origin_pattern:str = "grid" # "point" or "grid"
        class grid_pattern:
            env_spacing:float = 5.0
            x_offset:float = -20.0
            y_offset:float = -20.0
        class point_pattern:
            env_origins:List = [(0.0, 0.0, 0.0)]
#-----------------FLD Module--------------------
    class fld:
        latent_encoding_update_noise_level = 0.0
        latent_channel = 4
        observation_horizon = 31 # 51
        encoder_shape = [64, 64]
        decoder_shape = [64, 64]
        # use_pae = False
        # with_stand = False
        filter_latent = True # if filter the latent with a low pass filter
        filter_size = 30 # window size for filter
        one_hot = True
        one_hot_dim = 10
        state_idx_dict = {
            "base_pos": [0, 1, 2],
            "base_quat": [3, 4, 5, 6],
            "base_lin_vel": [7, 8, 9],
            "base_ang_vel": [10, 11, 12],
            "projected_gravity": [13, 14, 15],
            "dof_pos_leg_fl": [16, 17, 18],
            "dof_pos_leg_hl": [19, 20, 21],
            "dof_pos_leg_fr": [22, 23, 24],
            "dof_pos_leg_hr": [25, 26, 27],
            # "feet_pos_fl": [40, 41, 42],
            # "feet_pos_hl": [43, 44, 45],
            # "feet_pos_fr": [46, 47, 48],
            # "feet_pos_hr": [49, 50, 51],
        }
        load_root_pretrain = NAV_GYM_ROOT_DIR + "/resources/fld/pretrain/pae"
        load_fld_model = "model_20000.pt" 
    class task_sampler:
        name = "OfflineSamplerPAE"
        collect_samples = True
        collect_sample_step_interval = 5
        collect_elite_performance_threshold = 1.0
        library_size = 5000
        update_interval = 50
        elite_buffer_size = 1
        max_num_updates = 1000
        curriculum = True
        curriculum_scale = 1.25
        curriculum_performance_threshold = 0.8
        max_num_curriculum_updates = 5
        check_update_interval = 100
        class offline:
            pass
        class random:
            pass
        class gmm:
            num_components = 8
        class alp_gmm:
            init_num_components = 8
            min_num_components = 2
            max_num_components = 10
            random_type = "uniform" # "uniform" or "gmm"
        class classifier:
            enabled = False
            num_classes = 9

if __name__ == "__main__":
    env_cfg = LocomotionPAEEnvCfg()
    print(env_cfg.robot.init_state.dof_pos.items())