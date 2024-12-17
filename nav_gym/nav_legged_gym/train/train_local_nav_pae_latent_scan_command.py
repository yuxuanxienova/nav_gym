from nav_gym.nav_legged_gym.envs.locomotion_pae_latent_scan_command_env import LocomotionPAELatentScanEnv
from nav_gym.nav_legged_gym.envs.config_locomotion_pae_latent_scan_command_env import LocomotionPAELatentScanEnvCfg
from nav_gym.nav_legged_gym.envs.config_local_nav_pae_latent_scan_command_env import LocalNavPAEEnvCfg
from nav_gym.nav_legged_gym.envs.local_nav_pae_latent_scan_command_env import LocalNavPAEEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_local_nav_pae_latent_scan_command import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.nav_legged_gym.utils.config_utils import save_config_dict,save_config_py_file
import inspect
from nav_gym import NAV_GYM_ROOT_DIR
import torch
import os
import time
if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/local_nav/pae_latent_scan_command/" + time.strftime("%Y%m%d-%H%M%S"))
    # log_dir = None
    # checkpoint_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/20241103-205557/" + "model_600.pt")
    checkpoint_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs//local_nav/pae_latent_scan_command/cluster_1217_1/" + "model_6000.pt")
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)

    hl_env_cfg = LocalNavPAEEnvCfg()
    hl_env_cfg.env.num_envs = 512
    hl_env_cfg.gym.headless = True
    hl_env_cfg.ll_env_cfg.terrain_unity.grid_pattern.env_spacing = 8.0
    # hl_env_cfg.ll_env_cfg.terrain_unity.terrain_file = "/terrain/Plane1.obj"
    hl_env_cfg.ll_env_cfg.terrain_unity.translation = [-60.0, -60.0, 0.0]
    hl_env_cfg.ll_env_cfg.terrain_unity.terrain_file = "/terrain/LocomotionMap_v1.obj"

    env = LocalNavPAEEnv(hl_env_cfg, LocomotionPAELatentScanEnv)
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=log_dir, device="cuda:0")

    src_file_path = inspect.getfile(LocalNavPAEEnvCfg)
    dest_dir = os.path.join(log_dir, "config")
    save_config_py_file(src_file_path, dest_dir, dest_file_name = "LocomotionPAEEnvCfg.py")

    runner.load(checkpoint_dir)
    runner.learn(num_learning_iterations=train_cfg_dict["runner"]["max_iterations"], init_at_random_ep_len=True)
    runner.load(checkpoint_dir)