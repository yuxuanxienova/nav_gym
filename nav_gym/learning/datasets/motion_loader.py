from typing import List
from nav_gym import NAV_GYM_ROOT_DIR
from isaacgym.torch_utils import (
    quat_mul,
    quat_conjugate,
    normalize,
    quat_from_angle_axis,
    quat_rotate,
)
import os
import json
import torch

class MotionLoader:

    def __init__(self, device="cuda", file_names:List[str]=None,file_root:str=None, corruption_level=0.0, reference_observation_horizon=2, test_mode=False, test_observation_dim=None):
        self.device = device
        self.reference_observation_horizon = reference_observation_horizon
        if file_root is None:
            print("[MotionLoader] No motion file provided. Proceeding without loading any data.")
            exit(1)
        self.reference_state_idx_dict_file = os.path.join(file_root, "amp_state_idx_dict.json")
        with open(self.reference_state_idx_dict_file, 'r') as f:
            self.state_idx_dict = json.load(f)
        self.observation_dim = sum([ids[1] - ids[0] for state, ids in self.state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
        self.observation_start_dim = self.state_idx_dict["base_lin_vel"][0]
        data_list = []
        for file_name in file_names:
            file_path = os.path.join(file_root, file_name)
            loaded_data_i = torch.load(file_path, map_location=self.device)
            data_list.append(loaded_data_i)
        for i in range(len(data_list)):
            # Normalize and standardize quaternions
            base_quat = normalize(data_list[i][:, :, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]])
            base_quat[base_quat[:, :, -1] < 0] = -base_quat[base_quat[:, :, -1] < 0]
            data_list[i][:, :, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]] = base_quat
            #Data corruption
            data_list[i] = self._data_corruption(data_list[i], level=corruption_level)
        #motion_loader.data_list: [num_files]
        #motion_loader.data_list[i]: [num_motion, num_steps, motion_features_dim]
        self.data_list = data_list


    def _data_corruption(self, loaded_data, level=0):
        if level == 0:
            print(f"[MotionLoader] Proceeded without processing the loaded data.")
        else:
            loaded_data = self._rand_dropout(loaded_data, level)
            loaded_data = self._rand_noise(loaded_data, level)
            loaded_data = self._rand_interpolation(loaded_data, level)
            loaded_data = self._rand_duplication(loaded_data, level)
        return loaded_data

    def _rand_dropout(self, data, level=0):
        num_motion_clips, num_steps, reference_full_dim = data.size()
        num_dropouts = round(num_steps * level)
        if num_dropouts == 0:
            return data
        dropped_data = torch.zeros(num_motion_clips, num_steps - num_dropouts, reference_full_dim, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(num_motion_clips):
            step_ids = torch.randperm(num_steps)[:-num_dropouts].sort()[0]
            dropped_data[i] = data[i, step_ids]
        return dropped_data

    def _rand_interpolation(self, data, level=0):
        num_motion_clips, num_steps, reference_full_dim = data.size()
        num_interpolations = round((num_steps - 2) * level)
        if num_interpolations == 0:
            return data
        interpolated_data = data
        for i in range(num_motion_clips):
            step_ids = torch.randperm(num_steps)
            step_ids = step_ids[(step_ids != 0) * (step_ids != num_steps - 1)]
            step_ids = step_ids[:num_interpolations].sort()[0]
            interpolated_data[i, step_ids] = self.slerp(data[i, step_ids - 1], data[i, step_ids + 1], 0.5)
            interpolated_data[i, step_ids, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]] = self.quaternion_slerp(
                data[i, step_ids - 1, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]], 
                data[i, step_ids + 1, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]], 
                0.5
            )
        return interpolated_data

    def _rand_duplication(self, data, level=0):
        num_motion_clips, num_steps, reference_full_dim = data.size()
        num_duplications = round(num_steps * level) * 10
        if num_duplications == 0:
            return data
        duplicated_data = torch.zeros(num_motion_clips, num_steps + num_duplications, reference_full_dim, dtype=torch.float, device=self.device, requires_grad=False)
        step_ids = torch.randint(0, num_steps, (num_motion_clips, num_duplications), device=self.device)
        for i in range(num_motion_clips):
            duplicated_step_ids = torch.cat((torch.arange(num_steps, device=self.device), step_ids[i])).sort()[0]
            duplicated_data[i] = data[i, duplicated_step_ids]
        return duplicated_data

    def _rand_noise(self, data, level=0):
        noise_scales_dict = {
            "base_pos": 0.1,
            "base_quat": 0.01,
            "base_lin_vel": 0.1,
            "base_ang_vel": 0.2,
            "projected_gravity": 0.05,
            "base_height": 0.1,
            "dof_pos": 0.01,
            "dof_vel": 1.5
        }
        noise_scale_vec = torch.zeros_like(data[0, 0], device=self.device, dtype=torch.float, requires_grad=False)
        for key, value in self.state_idx_dict.items():
            noise_scale_vec[value[0]:value[1]] = noise_scales_dict[key] * level
        data += (2 * torch.randn_like(data) - 1) * noise_scale_vec
        return data
    
    def sample_k_steps_from_motion_clip(self, motion_id, k):
        num_steps = self.data_list[motion_id].size(1)
        #----Error handling----
        # Validate file_idx
        if motion_id < 0 or motion_id >= len(self.data_list):
            raise IndexError(f"[sample_k_steps_from_motion_clip] motion_id {motion_id} is out of bounds.")
        if num_steps < k:
            raise ValueError(f"[sample_k_steps_from_motion_clip] Motion clip has only {num_steps} steps, which is less than the requested {k} steps.")
        #--------------------------

        
        # Randomly select a starting index such that the window of k steps fits within the motion clip
        max_start_idx = num_steps - k
        start_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
        #motion_loader.data_list[i]: [num_motion, num_steps, motion_features_dim]
        sampled_steps = self.data_list[motion_id][0,start_idx:start_idx + k,:]  # Shape: [k, motion_features_dim]
        # sampled_steps : [k, motion_features_dim]
        return sampled_steps
        

    def slerp(self, value_low, value_high, blend):
        return (1.0 - blend) * value_low + blend * value_high

    def quaternion_slerp(self, quat_low, quat_high, blend):
        relative_quat = normalize(quat_mul(quat_high, quat_conjugate(quat_low)))
        angle = 2 * torch.acos(relative_quat[:, -1]).unsqueeze(-1)
        axis = normalize(relative_quat[:, :3])
        angle_slerp = self.slerp(torch.zeros_like(angle), angle, blend).squeeze(-1)
        relative_quat_slerp = quat_from_angle_axis(angle_slerp, axis)        
        return normalize(quat_mul(relative_quat_slerp, quat_low))

    def get_base_pos(self, motion_data_i):
        if "base_pos" in self.state_idx_dict:
            return motion_data_i[:, self.state_idx_dict["base_pos"][0]:self.state_idx_dict["base_pos"][1]]
        else:
            raise Exception("[MotionLoader] base_pos not specified in the state_idx_dict")

    def get_base_quat(self, motion_data_i):
        if "base_quat" in self.state_idx_dict:
            return motion_data_i[:, self.state_idx_dict["base_quat"][0]:self.state_idx_dict["base_quat"][1]]
        else:
            raise Exception("[MotionLoader] base_quat not specified in the state_idx_dict")

    def get_base_lin_vel(self, motion_data_i):
        if "base_lin_vel" in self.state_idx_dict:
            return motion_data_i[:, self.state_idx_dict["base_lin_vel"][0]:self.state_idx_dict["base_lin_vel"][1]]
        else:
            raise Exception("[MotionLoader] base_lin_vel not specified in the state_idx_dict")

    def get_base_ang_vel(self, motion_data_i):
        if "base_ang_vel" in self.state_idx_dict:
            return motion_data_i[:, self.state_idx_dict["base_ang_vel"][0]:self.state_idx_dict["base_ang_vel"][1]]
        else:
            raise Exception("[MotionLoader] base_ang_vel not specified in the state_idx_dict")

    def get_projected_gravity(self, motion_data_i):
        if "projected_gravity" in self.state_idx_dict:
            return motion_data_i[:, self.state_idx_dict["projected_gravity"][0]:self.state_idx_dict["projected_gravity"][1]]
        else:
            raise Exception("[MotionLoader] projected_gravity not specified in the state_idx_dict")

    def get_dof_pos(self, motion_data_i):
        if "dof_pos" in self.state_idx_dict:
            return motion_data_i[:, self.state_idx_dict["dof_pos"][0]:self.state_idx_dict["dof_pos"][1]]
        else:
            raise Exception("[MotionLoader] dof_pos not specified in the state_idx_dict")

    def get_dof_vel(self, motion_data_i):
        if "dof_vel" in self.state_idx_dict:
            return motion_data_i[:, self.state_idx_dict["dof_vel"][0]:self.state_idx_dict["dof_vel"][1]]
        else:
            raise Exception("[MotionLoader] dof_vel not specified in the state_idx_dict")

    def get_feet_pos(self, motion_data_i):
        if "feet_pos" in self.state_idx_dict:
            return motion_data_i[:, self.state_idx_dict["feet_pos"][0]:self.state_idx_dict["feet_pos"][1]]
        else:
            raise Exception("[MotionLoader] feet_pos not specified in the state_idx_dict")
        
    def compute_base_pos(self, motion_data_i, ori, dt):
        """approximate base position from base linear velocity"""
        base_lin_vel = quat_rotate(ori, self.get_base_lin_vel(motion_data_i))
        pos = base_lin_vel * dt
        return pos

    def compute_ori(self, motion_data_i, ori, dt):
        """approximate base orientation from base angular velocity"""
        base_ang_vel = quat_rotate(ori, self.get_base_ang_vel(motion_data_i))
        ori = (base_ang_vel * dt).squeeze(0)
        return ori[0], ori[1], ori[2]     
if __name__ == "__main__":
    datasets_root = os.path.join(NAV_GYM_ROOT_DIR + "/resources/fld/motion_data/")
    motion_name = "motion_data_pace1.0.pt"
    motion_path = os.path.join(datasets_root, motion_name)
    if not os.path.exists(motion_path):
        raise Exception(f"Could not find motion data at {motion_path}")
    motion_loader = MotionLoader(
        device="cuda",
        motion_file=motion_path,
        corruption_level=0.0,
        reference_observation_horizon=2,
        test_mode=False,
        test_observation_dim=None
    )

