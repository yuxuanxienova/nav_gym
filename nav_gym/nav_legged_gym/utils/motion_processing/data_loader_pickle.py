import pandas as pd
import numpy as np 
import torch


class MotionLoaderPickle:
    def __init__(self, device, motion_file=None, dim_of_interest=None, state_idx_dict=None):
        self.device = device
        self.motion_data = torch.load(motion_file, map_location=device)
        
        self.dim_of_interest = dim_of_interest
        self.state_idx_dict = state_idx_dict
    
    def get_base_pos(self):
        motion_idx = torch.tensor(self.state_idx_dict["base_pos"], device=self.device, dtype=torch.long, requires_grad=False)
        motion_data = self.motion_data[:, :, motion_idx]
        return motion_data
    
    def get_base_quat(self):
        motion_idx = torch.tensor(self.state_idx_dict["base_quat"], device=self.device, dtype=torch.long, requires_grad=False)
        motion_data = self.motion_data[:, :, motion_idx]
        return motion_data

    def get_base_lin_vel(self):
        motion_idx = torch.tensor(self.state_idx_dict["base_lin_vel"], device=self.device, dtype=torch.long, requires_grad=False)
        motion_data = self.motion_data[:, :, motion_idx]
        return motion_data
    
    def get_base_ang_vel(self):
        motion_idx = torch.tensor(self.state_idx_dict["base_ang_vel"], device=self.device, dtype=torch.long, requires_grad=False)
        motion_data = self.motion_data[:, :, motion_idx]
        return motion_data
    
    def get_projected_gravity(self):
        motion_idx = torch.tensor(self.state_idx_dict["projected_gravity"], device=self.device, dtype=torch.long, requires_grad=False)
        motion_data = self.motion_data[:, :, motion_idx]
        return motion_data
    
    def get_dof_pos(self):
        motion_idx = torch.tensor(self.state_idx_dict["dof_pos_leg_fr"] + self.state_idx_dict["dof_pos_leg_fl"] + \
                    self.state_idx_dict["dof_pos_leg_hr"] + self.state_idx_dict["dof_pos_leg_hl"], device=self.device, dtype=torch.long, requires_grad=False)
        motion_data = self.motion_data[:, :, motion_idx]
        return motion_data
    
    def get_dof_vel(self):
        motion_idx = torch.tensor(self.state_idx_dict["dof_vel_leg_fr"] + self.state_idx_dict["dof_vel_leg_fl"] + \
                    self.state_idx_dict["dof_vel_leg_hr"] + self.state_idx_dict["dof_vel_leg_hl"], device=self.device, dtype=torch.long, requires_grad=False)
        motion_data = self.motion_data[:, :, motion_idx]
        return motion_data
    
    def set_base_lin_vel(self, motion_data):
        motion_idx = torch.tensor(self.state_idx_dict["base_lin_vel"], device=self.device, dtype=torch.long, requires_grad=False)
        self.motion_data[:, :, motion_idx] = motion_data
        
    def set_base_ang_vel(self, motion_data):
        motion_idx = torch.tensor(self.state_idx_dict["base_ang_vel"], device=self.device, dtype=torch.long, requires_grad=False)
        self.motion_data[:, :, motion_idx] = motion_data