from nav_gym.nav_legged_gym.envs.legged_nav_env import LeggedNavEnv
from nav_gym.nav_legged_gym.envs.hl_nav_env_config_m import HLNavEnvCfg
from nav_gym.nav_legged_gym.envs.hierarchical_env_m import LocalNavEnv
from nav_gym.learning.runners.on_policy_modulized_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.runner_train_config_m_hl import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
import torch
import os
import time
if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)


    env = LocalNavEnv(HLNavEnvCfg(), LeggedNavEnv)
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=train_cfg_dict["runner"]["max_iterations"], init_at_random_ep_len=True)