
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.envs.locomotion_fld_env import LocomotionEnv
    ANY_ENV = Union[LocomotionEnv]

import torch
from nav_gym.learning.modules.samplers.offline import OfflineSampler
class FLDModule:
    def __init__(self,env:LocomotionEnv):
        #1. Parse environment arguments
        self.env_cfg = env.cfg
        self.fld_cfg = env.cfg.fld
        self.task_sampler_cfg=env.cfg.task_sampler
        self.fld_latent_channel = self.fld_cfg.latent_channel
        self.fld_observation_horizon = self.fld_cfg.observation_horizon
        self.fld_state_idx_dict = self.fld_cfg.state_idx_dict

        self.robot = env.robot
        self.base_pos = self.robot.root_pos_w
        self.base_quat = self.robot.root_quat_w
        self.base_lin_vel = self.robot.root_lin_vel_w
        self.base_ang_vel = self.robot.root_ang_vel_w
        self.projected_gravity = self.robot.projected_gravity_b
        self.dof_pos = self.robot.dof_pos
        self.default_dof_pos = self.robot.default_dof_pos
        self.dof_vel = self.robot.dof_vel

        self.terrain = env.terrain

        self.num_envs = env.num_envs
        self.device = env.device


        #2. other arguments
        self.decoded_obs_state_idx_dict = {}
        current_length = 0
        for state, ids in self.fld_state_idx_dict.items():
            if (state != "base_pos") and (state != "base_quat"):
                self.decoded_obs_state_idx_dict[state] = list(range(current_length, current_length + len(ids)))
                current_length = current_length + len(ids)
        #3. Initialize buffers
        self.fld_dim_of_interest = torch.cat([torch.tensor(ids, device=self.device, dtype=torch.long, requires_grad=False) for state, ids in self.fld_state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
        self.fld_observation_dim = len(self.fld_dim_of_interest)
        self.fld_observation_buf = torch.zeros(self.num_envs, self.fld_observation_horizon, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.fld_state = torch.zeros(self.num_envs, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.latent_encoding = torch.zeros(self.num_envs, self.fld_cfg.latent_channel, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.params = torch.zeros(self.num_envs, self.fld_cfg.latent_channel, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.latent_manifold = torch.zeros(self.num_envs, self.fld_cfg.latent_channel * 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.decoded_obs = torch.zeros(self.num_envs, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        #4. Initialize Task Sampler
        if self.task_sampler_cfg.name == "OfflineSampler":
            self.task_sampler =  OfflineSampler(self.device)
            self.task_sampler.load_data(self.fld_cfg.load_root+"/latent_params.pt")
        #5 load statistics
        statistics_dict = torch.load(fld_load_root + "/statistics.pt")
        self.state_transitions_mean, self.state_transitions_std = statistics_dict["state_transitions_mean"], statistics_dict["state_transitions_std"]
        self.latent_param_max, self.latent_param_min, self.latent_param_mean, self.latent_param_std = statistics_dict["latent_param_max"], statistics_dict["latent_param_min"], statistics_dict["latent_param_mean"], statistics_dict["latent_param_std"]
 
    def update(self):
        self._update_fld_observation_buf()
        self._update_latent_phase()
        pass
    def _update_fld_observation_buf(self):
        full_state = torch.cat(
            (
                self.robot.root_states[:, :3] - self.terrain.env_origins[:, :3],
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
                self.projected_gravity,
                self.dof_pos - self.default_dof_pos,
                self.dof_vel,
                # self.feet_pos,
                ), dim=1
            )
        for key, value in self.decoded_obs_state_idx_dict.items():
            self.fld_state[:, value] = full_state[:, self.fld_state_idx_dict[key]].clone()
        self.fld_observation_buf[:, :-1] = self.fld_observation_buf[:, 1:].clone()
        self.fld_observation_buf[:, -1] = self.fld_state.clone()
    def _update_latent_phase(self):
        #self.latent_encoding: Dim(num_envs,latent_channel,4)
        phase_t = self.latent_encoding[:, :, 0]#Dim(num_envs,latent_channel)
        frequency_t = self.latent_encoding[:, :, 1]#Dim(num_envs,latent_channel)
        amplitude_t = self.latent_encoding[:, :, 2]#Dim(num_envs,latent_channel)
        offset_t = self.latent_encoding[:, :, 3]#Dim(num_envs,latent_channel)

        plase_t = (frequency_t * self.dt + 0.5) % 1.0 - 0.5 #TODO: check if this is correct
        #---Add noise---
        # noise_level = self.fld_cfg.latent_encoding_update_noise_level
        # latent_param = self.latent_encoding[:, :, 1:].swapaxes(1, 2).flatten(1, 2)
        # latent_param += torch.randn_like(latent_param, device=self.device, dtype=torch.float, requires_grad=False) * self.latent_param_std * noise_level
        # self.latent_encoding[:, :, 1:] = latent_param.view(self.num_envs, 3, self.fld_latent_channel).swapaxes(1, 2)
        #----------------
        #amplitude_t.unsqueeze(-1): Dim(num_envs,latent_channel,1)
        reconstructed_z_tplus1 = amplitude_t.unsqueeze(-1) * torch.sin(2 * torch.pi * (frequency_t.unsqueeze(-1) * self.fld.args + phase_t.unsqueeze(-1))) + offset_t.unsqueeze(-1)#TODO: what is self.fld.args???
        #reconstructed_z_tplus1: Dim(num_envs,latent_channel,horizon_length)
        with torch.no_grad():
            #reconstructed_z_tplus1: Dim(num_envs,latent_channel,horizon_length)
            decoded_obs_buf_pred = self.fld.decoder(reconstructed_z_tplus1)
            #decoded_obs_buf_pred: Dim(num_envs,input_channel,horizon_length)
        decoded_obs_buf_raw = decoded_obs_buf_pred.swapaxes(1, 2)#Dim(num_envs,horizon_length,input_channel)
        self.decoded_obs[:] = decoded_obs_buf_raw[:, -1, :] * self.state_transitions_std + self.state_transitions_mean#Dim(num_envs,input_channel)
        # if self.cfg.fld.with_stand:
        #     self.decoded_obs[:] = self.decoded_obs * (1 - self.standing_latent) + self.standing_latent * self.standing_obs
        self.latent_manifold[:] = torch.hstack(
            (
                amplitude_t * torch.sin(2.0 * torch.pi * phase_t) + offset_t,
                amplitude_t * torch.cos(2.0 * torch.pi * phase_t) + offset_t,
                )
            )
    
    def sample_latent_encoding(self,env_ids):
        if len(env_ids) == 0:
            return
        self.latent_encoding[env_ids, :, 0] = torch.rand((len(env_ids), self.fld_latent_channel),device=self.device) * 1.0 - 0.5  # Scaling to range [-0.5, 0.5]
        self.latent_encoding[env_ids, :, 1:] = self.task_sampler.sample(len(env_ids)).view(len(env_ids), 3, self.fld_latent_channel).swapaxes(1, 2)
        # if self.fld_cfg.with_stand:
        #     # 20% chance of standing
        #     self.standing_latent[env_ids, :] = (torch.randint(0, 5, (len(env_ids), 1), device=self.device, dtype=torch.float, requires_grad=False) == 0).float()
