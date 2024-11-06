from nav_gym import NAV_GYM_ROOT_DIR
from isaacgym.torch_utils import (
    quat_rotate_inverse,
)
import os
import json
import torch

class AMPRecorder:

    def __init__(self, env, num_steps, amp_state_idx_dict=None, ignore_first_num_steps=100, motion_file=None):
        self.env = env
        self.num_steps = num_steps
        self.ignore_first_num_steps = ignore_first_num_steps
        if motion_file is None:
            motion_file = LEGGED_GYM_ROOT_DIR + "/resources/robots/anymal_c/datasets/motion_data.pt"
        self.motion_file = motion_file
        self.amp_state_idx_dict_file = os.path.join(os.path.dirname(motion_file), "amp_state_idx_dict.json")
        if amp_state_idx_dict is None:
            if getattr(self.env, "amp_state_idx_dict", None) is None:
                raise Exception("[MotionRecorder] No AMP state index dictionary provided")
            else:
                self.amp_state_idx_dict = env.amp_state_idx_dict
                self.amp_full_dim = self.env.amp_full_dim
        else:
            self.amp_state_idx_dict = amp_state_idx_dict
            self.amp_full_dim = sum([ids[1] - ids[0] for ids in self.amp_state_idx_dict.values()])
        self.amp_states = torch.zeros(
            env.num_envs, self.amp_full_dim, dtype=torch.float, device=env.device, requires_grad=False
        )
        self.data = torch.zeros(
            env.num_envs, self.num_steps, self.amp_full_dim, dtype=torch.float, device=env.device, requires_grad=False
        )
        self.stop_recording = False

    def record(self, step):
        if step == 0:
            print(f"[MotionRecorder] Start recording")
        if self.stop_recording:
            pass
        else:
            amp_record_states = self._get_amp_record_states()
            self.data[:, step, :] = amp_record_states
            print(f"[MotionRecorder] Process {step} / {self.num_steps} steps")
            if step >= self.num_steps - 1:
                torch.save(self.data[:, self.ignore_first_num_steps:, :], self.motion_file)
                print(f"[MotionRecorder] Saved {self.env.num_envs} motion clips in {self.motion_file} while ignoring first {self.ignore_first_num_steps} steps")
                with open(self.amp_state_idx_dict_file, 'w') as f:
                    json.dump(self.amp_state_idx_dict, f)
                self.stop_recording = True

    def _get_amp_record_states(self):
        for key, value in self.amp_state_idx_dict.items():
            if key == "base_pos":
                self.amp_states[:, value[0]: value[1]] = self._get_base_pos()
            elif key == "base_quat":
                self.amp_states[:, value[0]: value[1]] = self._get_base_quat()
            # elif key == "base_lin_vel":
            #     self.amp_states[:, value[0]: value[1]] = self._get_base_lin_vel()
            # elif key == "base_ang_vel":
            #     self.amp_states[:, value[0]: value[1]] = self._get_base_ang_vel()
            # elif key == "dof_pos":
            #     self.amp_states[:, value[0]: value[1]] = self._get_dof_pos()
            # elif key == "dof_vel":
            #     self.amp_states[:, value[0]: value[1]] = self._get_dof_vel()
            # elif key == "projected_gravity":
            #     self.amp_states[:, value[0]: value[1]] = self._get_projected_gravity()
            elif key == "feet_pos":
                self.amp_states[:, value[0]: value[1]] = self._get_feet_pos()
            else:
                self.amp_states[:, value[0]: value[1]] = getattr(self.env, key)
        return self.amp_states
    
    def _get_base_pos(self):
        return self.env.root_states[:, :3] - self.env.env_origins[:, :3]
    
    def _get_base_quat(self):
        return self.env.root_states[:, 3:7]
    
    # def _get_base_lin_vel(self):
    #     return self.env.root_states[:, 7:10]
    
    # def _get_base_ang_vel(self):
    #     return self.env.base_ang_vel[:]
    
    # def _get_dof_pos(self):
    #     return self.env.dof_pos[:]
    
    # def _get_dof_vel(self):
    #     return self.env.dof_vel[:]
    
    # def _get_projected_gravity(self):
    #     return self.env.projected_gravity[:]

    def _get_feet_pos(self):
        feet_pos_global = self.env.rigid_body_pos[:, self.env.feet_indices, :3]
        feet_pos_local = torch.zeros_like(feet_pos_global)
        for i in range(len(self.env.feet_indices)):
            feet_pos_local[:, i] = quat_rotate_inverse(
                self.env.base_quat,
                feet_pos_global[:, i]
            )
        return feet_pos_local.flatten(1, 2)