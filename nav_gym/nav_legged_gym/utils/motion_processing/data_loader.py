import pandas as pd
from nav_gym import NAV_GYM_ROOT_DIR
import numpy as np 
import torch


class MotionLoader:
    def __init__(self, device, motion_file=None):
        self.device = device
        if motion_file is None:
            motion_file = NAV_GYM_ROOT_DIR + "/resources/mocaps/Dingo/motionData_dingo_D1_ex03_KAN02_013_anymal.txt"
        motion_df = pd.read_csv(motion_file, sep=", ")
        self.motion_df = motion_df
        self.state_idx_dict = {
            "base_pos": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "base_position" in c], 
            "base_quat": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "base_quaternion" in c],
            "base_lin_vel": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "base_linearvelocityInBase" in c],
            "base_ang_vel": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "base_angularvelocityInBase" in c],
            # "projected_gravity": [13, 14, 15],
            "dof_pos_leg_fr": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "jointAngle_RF" in c],
            "dof_pos_leg_fl": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "jointAngle_LF" in c],
            "dof_pos_leg_hr": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "jointAngle_RH" in c],
            "dof_pos_leg_hl": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "jointAngle_LH" in c],
            "dof_vel_leg_fr": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "jointVelocity_RF" in c],
            "dof_vel_leg_fl": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "jointVelocity_LF" in c],
            "dof_vel_leg_hr": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "jointVelocity_RH" in c],
            "dof_vel_leg_hl": [motion_df.columns.get_loc(c) for c in motion_df.keys() if "jointVelocity_LH" in c],
        }
    
    # def get_motion_length(self):
    #     return 

    def get_motion_data(self):
        """Get the motion data based on dictionary from motion dataframe"""
        motion_idx = [x for xs in list(self.state_idx_dict.values()) for x in xs]
        motion_data = self.motion_df.iloc[:, motion_idx].values
        return torch.tensor(motion_data, device=self.device, dtype=torch.float, requires_grad=False)
    
    def get_base_pos(self):
        motion_idx = self.state_idx_dict["base_pos"]
        motion_data = self.motion_df.iloc[:, motion_idx].values
        return torch.tensor(motion_data, device=self.device, dtype=torch.float, requires_grad=False)
    
    def get_base_quat(self):
        motion_idx = self.state_idx_dict["base_quat"]
        motion_data = self.motion_df.iloc[:, motion_idx].values # wxyz
        # convert to the isaacgym quaternion convention (from wxyz to xyzw)
        new_motion_data = np.zeros_like(motion_data)
        new_motion_data[:, :3] = motion_data[:, 1:].copy()
        new_motion_data[:, -1] = motion_data[:, 0].copy() # xyzw
        return torch.tensor(new_motion_data, device=self.device, dtype=torch.float, requires_grad=False)

    def get_base_lin_vel(self):
        motion_idx = self.state_idx_dict["base_lin_vel"]
        motion_data = self.motion_df.iloc[:, motion_idx].values
        return torch.tensor(motion_data, device=self.device, dtype=torch.float, requires_grad=False)
    
    def get_base_ang_vel(self):
        motion_idx = self.state_idx_dict["base_ang_vel"]
        motion_data = self.motion_df.iloc[:, motion_idx].values
        return torch.tensor(motion_data, device=self.device, dtype=torch.float, requires_grad=False)
    
    def get_dof_pos(self):
        motion_idx = self.state_idx_dict["dof_pos_leg_fr"] + self.state_idx_dict["dof_pos_leg_fl"] + \
                    self.state_idx_dict["dof_pos_leg_hr"] + self.state_idx_dict["dof_pos_leg_hl"]
        motion_data = self.motion_df.iloc[:, motion_idx].values
        return torch.tensor(motion_data, device=self.device, dtype=torch.float, requires_grad=False)
    
    def get_dof_vel(self):
        motion_idx = self.state_idx_dict["dof_vel_leg_fr"] + self.state_idx_dict["dof_vel_leg_fl"] + \
                    self.state_idx_dict["dof_vel_leg_hr"] + self.state_idx_dict["dof_vel_leg_hl"]
        motion_data = self.motion_df.iloc[:, motion_idx].values
        return torch.tensor(motion_data, device=self.device, dtype=torch.float, requires_grad=False)
