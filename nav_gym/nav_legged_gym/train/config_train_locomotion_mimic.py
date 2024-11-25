from nav_gym.learning.modules.config.model_config import ArchitectureCfg, GaussianDistributionCfg,FLDLocomotionActorCriticCfg, MimicLocomotionActorCriticCfg
from typing import List, Tuple, Union, Optional

class TrainConfig:
    seed = 1
    runner_class_name = 'OnPolicyModulizedRunner'
    class actor_critic:
        class_name:str = "ActorCriticSeparate"
        actor_architecture = MimicLocomotionActorCriticCfg()
        critic_architecture = MimicLocomotionActorCriticCfg()
        action_distribution = GaussianDistributionCfg()

        
    class algorithm:
        # training params
        value_loss_coef = 0.5
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0075
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.992
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 #24 per iteration
        max_iterations = 500000 # number of policy updates
        empirical_normalization: bool = True

        # logging
        save_interval = 300 # check for potential saves every this many iterations
        experiment_name = 'local_nav'
        run_name: Union[int, str] = ""
        # load and resume
        resume: bool = False
        load_run: Union[int, str] = -1  # -1 = last run
        checkpoint: int = -1  # -1 = last saved model
        resume_path: Optional[str] = None  # updated from load_run and chkpt

        logger: str = "tensorboard"  # tensorboard, wandb, or neptune
        # wandb_project: str = "legged_gym"
        # neptune_project: str = "legged_gym"