from nav_gym.learning.datasets.motion_loader import MotionLoader
import os 
from nav_gym import NAV_GYM_ROOT_DIR
import random
import torch
class AMPDemoStorage:
    def __init__(self):
        #Load the Motion Data
        self.datasets_root = os.path.join(NAV_GYM_ROOT_DIR + "/resources/fld/motion_data/")
        self.motion_names = ["motion_data_pace1.0.pt","motion_data_walk01_0.5.pt","motion_data_walk03_0.5.pt","motion_data_canter02_1.5.pt"]
        self.num_files = len(self.motion_names)

        self.motion_loader = MotionLoader(
            device="cuda",
            file_names=self.motion_names,
            file_root=self.datasets_root,
            corruption_level=0.0,
            reference_observation_horizon=2,
            test_mode=False,
            test_observation_dim=None
        )
        self.state_idx_dict = self.motion_loader.state_idx_dict

    def sample_amp_demo_obs_batch(self, sample_length, batch_size=1):
        for i in range(batch_size):
            amp_demo_obs = self.sample_amp_demo_obs_single(sample_length)
            if i == 0:
                amp_demo_obs_batch = amp_demo_obs.unsqueeze(0)# [1, k, amp_obs_dim]
            else:
                amp_demo_obs_batch = torch.cat([amp_demo_obs_batch, amp_demo_obs], dim=0)
        # amp_demo_obs_batch : [batch_size, k, amp_obs_dim]
        return amp_demo_obs_batch
    def sample_amp_demo_obs_single(self, sample_length):
        motion_idx = random.randint(0, self.num_files - 1)
        sampled_steps = self.motion_loader.sample_k_steps_from_motion_clip(motion_idx, sample_length)
        # sampled_steps : [k, motion_features_dim]
        sampled_dof_pos = sampled_steps[:, self.state_idx_dict["dof_pos"][0]:self.state_idx_dict["dof_pos"][1]]# Shape: [k, 12]
        sampled_dof_vel = sampled_steps[:, self.state_idx_dict["dof_vel"][0]:self.state_idx_dict["dof_vel"][1]]# Shape: [k, 12]
        # amp_demo_obs : [k, amp_obs_dim]
        amp_demo_obs = torch.cat([sampled_dof_pos, sampled_dof_vel], dim=1)
        return amp_demo_obs
    
if __name__ == "__main__":
    amp_demo_storage = AMPDemoStorage()
    amp_demo_obs = amp_demo_storage.sample_amp_demo_obs_batch(16)
    print(f"AMP Demo Observation Shape: {amp_demo_obs.shape}")
    print(f"AMP Demo Observation: {amp_demo_obs}")