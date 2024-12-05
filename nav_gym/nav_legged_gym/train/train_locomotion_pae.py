from nav_gym.nav_legged_gym.envs.config_locomotion_pae_env import LocomotionPAEEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_pae_env import LocomotionPAEEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion_pae import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.nav_legged_gym.utils.config_utils import save_config_dict,save_config_py_file
from nav_gym import NAV_GYM_ROOT_DIR
import inspect
import torch
import os
import time
if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/locomotion_pae/" + time.strftime("%Y%m%d-%H%M%S"))
    # checkpoint_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/20241108-171530/" + "model_2400.pt")
    log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)

    env_cfg = LocomotionPAEEnvCfg()
    env_cfg.env.num_envs = 1
    # env_cfg.env.num_envs = 4096
    env_cfg.gym.headless = True
    env_cfg.terrain_unity.translation = [0.0, 0.0, -1.0]
    # env_cfg.terrain_unity.env_origin_pattern = "point"
    env_cfg.gym.viewer.eye = (3.0, 3.0, 3.0)
    
    # src_file_path = inspect.getfile(LocomotionPAEEnvCfg)
    # dest_dir = os.path.join(log_dir, "config")
    # save_config_py_file(src_file_path, dest_dir, dest_file_name = "LocomotionPAEEnvCfg.py")

    env = LocomotionPAEEnv(env_cfg)
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=log_dir, device="cuda:0")
    # runner.load(checkpoint_dir)
    runner.learn(num_learning_iterations=train_cfg_dict["runner"]["max_iterations"], init_at_random_ep_len=True)