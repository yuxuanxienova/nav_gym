from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
from nav_gym.nav_legged_gym.envs.config_local_nav_env import LocalNavEnvCfg
from nav_gym.nav_legged_gym.envs.local_nav_env import LocalNavEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_local_nav import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym import NAV_GYM_ROOT_DIR
import torch
import os
import time
if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/local_nav/" + time.strftime("%Y%m%d-%H%M%S"))
    # log_dir = None
    # checkpoint_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/20241103-205557/" + "model_600.pt")
    # checkpoint_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/20241107-201846/" + "model_900.pt")
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)


    env = LocalNavEnv(LocalNavEnvCfg(), LocomotionEnv)
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=log_dir, device="cuda:0")
    # runner.load(checkpoint_dir)
    runner.learn(num_learning_iterations=train_cfg_dict["runner"]["max_iterations"], init_at_random_ep_len=True)