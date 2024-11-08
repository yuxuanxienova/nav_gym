from nav_gym.nav_legged_gym.envs.config_locomotion_fld_env import LocomotionFLDEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_fld_env import LocomotionFLDEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
import torch
import os
import time
if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    # checkpoint_dir = os.path.join(os.path.dirname(__file__), "logs/20241029-120802/" + "model_4200.pt")
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)


    env = LocomotionFLDEnv(LocomotionFLDEnvCfg())
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=log_dir, device="cuda:0")
    # runner.load(checkpoint_dir)
    runner.learn(num_learning_iterations=train_cfg_dict["runner"]["max_iterations"], init_at_random_ep_len=True)