from nav_gym.nav_legged_gym.envs.config_locomotion_mimic_env import LocomotionMimicEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_mimic_env import LocomotionMimicEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion_mimic import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.nav_legged_gym.utils.config_utils import config_to_dict
from nav_gym.nav_legged_gym.utils.config_utils import save_config
import torch
import os
import time
if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "logs/locomotion_mimic/" + time.strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "logs/locomotion_mimic/20241125-181423/" + "model_10500.pt")
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)

    env_cfg = LocomotionMimicEnvCfg()
    log_dir_env_cfg = log_dir + "/env_cfg.json"
    os.makedirs(log_dir_env_cfg, exist_ok=True)
    save_config(env_cfg, log_dir + "/env_cfg.json")

    env = LocomotionMimicEnv(env_cfg)
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=log_dir, device="cuda:0")
    runner.load(checkpoint_dir)
    runner.learn(num_learning_iterations=train_cfg_dict["runner"]["max_iterations"], init_at_random_ep_len=True)