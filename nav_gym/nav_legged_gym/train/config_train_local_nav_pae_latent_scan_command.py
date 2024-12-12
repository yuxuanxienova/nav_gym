from nav_gym.learning.modules.config.model_config import TeacherArchCfg, betaActionDistCfg, betaActionDistCfg_pae_latent, GaussianDistributionCfg
from typing import List, Tuple, Union, Optional

class TrainConfig:
    seed = 1
    runner_class_name = 'OnPolicyModulizedRunner'
    # class policy:
    #     init_noise_std = 1.0
    #     actor_hidden_dims = [128, 64, 32 ]#
    #     critic_hidden_dims = [128, 64, 32]#
    #     activation = 'relu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    #     # only for 'ActorCriticRecurrent':
    #     # rnn_type = 'lstm'
    #     # rnn_hidden_size = 512
    #     # rnn_num_layers = 1
    class actor_critic:
        class_name:str = "ActorCriticSeparate"
        actor_architecture = TeacherArchCfg()
        critic_architecture = TeacherArchCfg()
        # action_distribution = betaActionDistCfg_pae_latent()
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
        max_iterations = 15000 # number of policy updates
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