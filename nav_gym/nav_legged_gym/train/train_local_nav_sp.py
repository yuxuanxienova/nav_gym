from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
from nav_gym.nav_legged_gym.envs.config_local_nav_env import LocalNavEnvCfg
from nav_gym.nav_legged_gym.envs.local_nav_env import LocalNavEnv
from nav_gym.learning.runners.on_policy_runner_sp import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_local_nav import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
import torch
import os
import time
if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    # log_dir = None
    # checkpoint_dir = os.path.join(os.path.dirname(__file__), "logs/20241103-205557/" + "model_600.pt")
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)


    env = LocalNavEnv(LocalNavEnvCfg(), LocomotionEnv)
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=log_dir, device="cuda:0")
    # runner.load(checkpoint_dir)
    runner.learn(num_learning_iterations=train_cfg_dict["runner"]["max_iterations"], init_at_random_ep_len=True)