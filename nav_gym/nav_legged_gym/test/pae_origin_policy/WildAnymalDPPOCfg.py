
from nav_gym.nav_legged_gym.common.assets.robots.legged_robots.legged_robots_cfg import LeggedRobotCfg,anymal_d_robot_cfg,anymal_d_robot_pae_cfg
from nav_gym.nav_legged_gym.common.gym_interface.gym_interface_cfg import GymInterfaceCfg, ViewerCfg,SimParamsCfg,PhysxCfg
from nav_gym.nav_legged_gym.common.sensors.sensors_cfg import RaycasterCfg,OmniScanRaycasterCfg,FootScanCfg,GridPatternCfg,BaseScanCfg
import nav_gym.nav_legged_gym.common.observations.observations as O
import nav_gym.nav_legged_gym.common.rewards.rewards as R
import nav_gym.nav_legged_gym.common.terminations.terminations as T
import nav_gym.nav_legged_gym.common.curriculum.curriculum as C
from typing import Dict, List, Tuple, Union
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.nav_legged_gym.common.commands.commands_cfg import UnifromVelocityCommandCfg
from nav_gym import NAV_GYM_ROOT_DIR
from nav_gym.nav_legged_gym.utils.config_utils import configclass

@configclass
class PolicyCfg:
    init_noise_std: float = 1.0
    actor_hidden_dims: Tuple = (512, 256, 128)
    critic_hidden_dims: Tuple = (512, 256, 128)
    activation: str = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    # only for 'ActorCriticRecurrent':
    rnn_type = 'lstm'
    rnn_hidden_size = 512
    rnn_num_layers = 1

@configclass
class AlgorithmCfg:
    # training params
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.001  # default 0.01
    entropy_decay: float = 1.0   # decaying factor for entropy
    num_learning_epochs: int = 5
    num_mini_batches: int = 4  # mini batch size = num_envs * nsteps / nminibatches
    learning_rate: float = 1.0e-3  # 5.e-4, tune
    schedule: str = "adaptive"  # adaptive, fixed
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01 # tune
    max_grad_norm: float = 1.0
    max_act_std: float = 1.0

@configclass
class RunnerCfg:
    policy_class_name: str = "ActorCritic"
    algorithm_class_name: str = "PPO"
    num_steps_per_env: int = 24  # per iteration
    max_iterations: int = 1500  # number of policy updates
    empirical_normalization: bool = True
    # curriculum training for high-level policy
    curriculum = False
    curriculum_iter = 0
    asymmetric = False

    # logging
    save_interval: int = 50  # check for potential saves every this many iterations
    experiment_name: str = "test"
    run_name: Union[int, str] = ""
    # load and resume
    resume: bool = False
    load_run: Union[int, str] = -1  # -1 = last run
    checkpoint: int = -1  # -1 = last saved model
    resume_path: str = None  # updated from load_run and chkpt

@configclass
class PPOCfg:
    seed: int = 1
    runner_class_name: str = "OnPolicyRunner"
    policy: PolicyCfg = PolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    runner: RunnerCfg = RunnerCfg()

@configclass
class WildAnymalDPPOCfg(PPOCfg):
    runner_class_name = "FLDOnPolicyRunner"
    policy = PolicyCfg(
        init_noise_std = 0.5,
        # actor_hidden_dims = (256, 160, 128),
        # critic_hidden_dims = (256, 160, 128),
        actor_hidden_dims = (512, 256, 128),
        critic_hidden_dims = (512, 256, 128),
    )
    runner = RunnerCfg(
        run_name="min_forces",
        experiment_name="flat_wild_anymal_true_dof",
        load_run=-1,
        algorithm_class_name = "PPO", 
        policy_class_name = "ActorCritic",
        max_iterations=30000,
        num_steps_per_env=36,
        # empirical_normalization=False
    )
    algorithm = AlgorithmCfg(entropy_coef=0.001, max_act_std=1.0)
    runner.task_sampler_class_name = "OfflineSampler" # "OfflineSampler", "GMMSampler", "RandomSampler", "ALPGMMSampler"