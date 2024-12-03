

import torch
from nav_gym.learning.modules.samplers.offline import OfflineSampler
from nav_gym.learning.modules.samplers.gmm import GMMSampler
from nav_gym import NAV_GYM_ROOT_DIR
from nav_gym.learning.modules.fld.fld import FLD
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.envs.locomotion_fld_env import LocomotionFLDEnv
    ANY_ENV = Union[LocomotionFLDEnv]
from collections import OrderedDict

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            # Remove 'module.' prefix
            name = k[len('module.'):]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

class FLDModule:
    def __init__(self,env:"ANY_ENV"):
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
        self.base_lin_vel = self.robot.root_lin_vel_w#TODO check if this is correct .root_lin_vel_b??
        self.base_ang_vel = self.robot.root_ang_vel_w
        self.projected_gravity = self.robot.projected_gravity_b
        self.dof_pos = self.robot.dof_pos
        self.default_dof_pos = self.robot.default_dof_pos
        self.dof_vel = self.robot.dof_vel

        self.terrain = env.terrain

        self.num_envs = env.num_envs
        self.device = env.device

        self.dt = env.dt
        #2. other arguments
        self.target_fld_state_state_idx_dict = {}
        current_length = 0
        for state, ids in self.fld_state_idx_dict.items():
            if (state != "base_pos") and (state != "base_quat"):
                self.target_fld_state_state_idx_dict[state] = list(range(current_length, current_length + len(ids)))
                current_length = current_length + len(ids)
        #3. Initialize buffers
        self.fld_dim_of_interest = torch.cat([torch.tensor(ids, device=self.device, dtype=torch.long, requires_grad=False) for state, ids in self.fld_state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
        self.fld_observation_dim = len(self.fld_dim_of_interest)
        # self.fld_observation_buf = torch.zeros(self.num_envs, self.fld_observation_horizon, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.fld_state = torch.zeros(self.num_envs, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.latent_encoding = torch.zeros(self.num_envs, self.fld_cfg.latent_channel, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.params = torch.zeros(self.num_envs, self.fld_cfg.latent_channel, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.latent_manifold = torch.zeros(self.num_envs, self.fld_cfg.latent_channel * 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_pred = torch.zeros(self.num_envs, self.fld_observation_horizon,self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_fld_state = torch.zeros(self.num_envs, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        #4. Initialize Task Sampler
        if self.task_sampler_cfg.name == "OfflineSampler":
            self.task_sampler =  OfflineSampler(self.device)
            self.task_sampler.load_data(self.fld_cfg.load_root_pretrain+"/latent_params.pt")
        #5 load fld model
        self.fld = FLD(self.fld_observation_dim, self.fld_observation_horizon, self.fld_latent_channel, self.device, encoder_shape=self.fld_cfg.encoder_shape, decoder_shape=self.fld_cfg.decoder_shape).eval()
        fld_load_root = self.fld_cfg.load_root_pretrain
        fld_load_model = self.fld_cfg.load_fld_model
        loaded_dict = torch.load(fld_load_root + "/" + fld_load_model)
        #If the model was saved with DataParallel, need to remove the 'module.' prefix
        # Assuming state dict is under 'fld_state_dict' key
        original_state_dict = loaded_dict['fld_state_dict']
        clean_state_dict = remove_module_prefix(original_state_dict)
        # Remove 'args' and 'freqs' keys
        keys_to_remove = ['args', 'freqs']
        for key in keys_to_remove:
            if key in clean_state_dict:
                del clean_state_dict[key]
        self.fld.load_state_dict(clean_state_dict)
        self.fld.eval()
        statistics_dict = torch.load(fld_load_root + "/statistics.pt")
        self.state_transitions_mean, self.state_transitions_std = statistics_dict["state_transitions_mean"], statistics_dict["state_transitions_std"]
        self.latent_param_max, self.latent_param_min, self.latent_param_mean, self.latent_param_std = statistics_dict["latent_param_max"], statistics_dict["latent_param_min"], statistics_dict["latent_param_mean"], statistics_dict["latent_param_std"]
 
    def on_env_post_physics_step(self):
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
                self.dof_pos - self.default_dof_pos,#TODO check if this is correct
                self.dof_vel,
                # self.feet_pos,
                ), dim=1
            )
        for key, value in self.target_fld_state_state_idx_dict.items():
            self.fld_state[:, value] = full_state[:, self.fld_state_idx_dict[key]].clone()
        # self.fld_observation_buf[:, :-1] = self.fld_observation_buf[:, 1:].clone()
        # self.fld_observation_buf[:, -1] = self.fld_state.clone()
    def _update_latent_phase(self):
        #self.latent_encoding: Dim(num_envs,latent_channel,4)
        phase_t = self.latent_encoding[:, :, 0]#Dim(num_envs,latent_channel)
        frequency_t = self.latent_encoding[:, :, 1]#Dim(num_envs,latent_channel)
        amplitude_t = self.latent_encoding[:, :, 2]#Dim(num_envs,latent_channel)
        offset_t = self.latent_encoding[:, :, 3]#Dim(num_envs,latent_channel)
        #one step forward in phase
        phase_tplus1 = (phase_t + frequency_t * self.dt + 0.5) % 1.0 - 0.5 #TODO: check if this is correct
        #---Add noise---
        noise_level = self.fld_cfg.latent_encoding_update_noise_level
        latent_param = self.latent_encoding[:, :, 1:].swapaxes(1, 2).flatten(1, 2)
        latent_param = latent_param + torch.randn_like(latent_param, device=self.device, dtype=torch.float, requires_grad=False) * self.latent_param_std * noise_level
        #----------------
        #Update latent encoding
        self.latent_encoding[:, :, 0] = phase_tplus1
        self.latent_encoding[:, :, 1:] = latent_param.view(self.num_envs, 3, self.fld_latent_channel).swapaxes(1, 2)
        #amplitude_t.unsqueeze(-1): Dim(num_envs,latent_channel,1)
        reconstructed_z_tplus1 = amplitude_t.unsqueeze(-1) * torch.sin(2 * torch.pi * (frequency_t.unsqueeze(-1) * self.fld.args + phase_tplus1.unsqueeze(-1))) + offset_t.unsqueeze(-1)#TODO: what is self.fld.args???
        #reconstructed_z_tplus1: Dim(num_envs,latent_channel,horizon_length)
        with torch.no_grad():
            #reconstructed_z_tplus1: Dim(num_envs,latent_channel,horizon_length)
            target_fld_state_buf_pred = self.fld.decoder(reconstructed_z_tplus1)
            #target_fld_state_buf_pred: Dim(num_envs,input_channel,horizon_length)
        target_fld_state_buf_raw = target_fld_state_buf_pred.swapaxes(1, 2)#Dim(num_envs,horizon_length,input_channel)
        self.obs_pred = target_fld_state_buf_raw * self.state_transitions_std.reshape(1,1,-1) + self.state_transitions_mean.reshape(1,1,-1)
        self.target_fld_state[:] = target_fld_state_buf_raw[:, -1, :] * self.state_transitions_std + self.state_transitions_mean#Dim(num_envs,input_channel)
        # if self.cfg.fld.with_stand:
        #     self.target_fld_state[:] = self.target_fld_state * (1 - self.standing_latent) + self.standing_latent * self.standing_obs
        self.latent_manifold[:] = torch.hstack(
            (
                amplitude_t * torch.sin(2.0 * torch.pi * phase_t) + offset_t,
                amplitude_t * torch.cos(2.0 * torch.pi * phase_t) + offset_t,
                )
            )
    def on_env_reset_idx(self,env_ids):
        self._sample_latent_encoding(env_ids)
    def on_env_resample_commands(self,env_ids):
        self._sample_latent_encoding(env_ids)
    def _sample_latent_encoding(self,env_ids):
        if len(env_ids) == 0:
            return
        self.latent_encoding[env_ids, :, 0] = torch.rand((len(env_ids), self.fld_latent_channel),device=self.device) * 1.0 - 0.5  # Scaling to range [-0.5, 0.5]
        self.latent_encoding[env_ids, :, 1:] = self.task_sampler.sample(len(env_ids)).view(len(env_ids), 3, self.fld_latent_channel).swapaxes(1, 2)
        
        print("[INFO] Sampled latent encoding for env_ids: ", env_ids)
        print("[INFO] self.latent_encoding[env_ids, :, 0]: ", self.latent_encoding[env_ids, :, 0])
        print("[INFO] self.latent_encoding[env_ids, :, 1]: ", self.latent_encoding[env_ids, :, 1])
        print("[INFO] self.latent_encoding[env_ids, :, 2]: ", self.latent_encoding[env_ids, :, 2])
        print("[INFO] self.latent_encoding[env_ids, :, 3]: ", self.latent_encoding[env_ids, :, 3])
        # if self.fld_cfg.with_stand:
        #     # 20% chance of standing
        #     self.standing_latent[env_ids, :] = (torch.randint(0, 5, (len(env_ids), 1), device=self.device, dtype=torch.float, requires_grad=False) == 0).float()

    def get_reconstructed_base_lin_vel(self):
        #return: Dim(num_envs,prediction_horizon,3)
        return self.obs_pred[:,:,torch.tensor(self.target_fld_state_state_idx_dict["base_lin_vel"], device=self.device, dtype=torch.long, requires_grad=False)]
        
    def get_reconstructed_base_ang_vel(self):
        #return: Dim(num_envs,prediction_horizon,3)
        return self.obs_pred[:,:, torch.tensor(self.target_fld_state_state_idx_dict["base_ang_vel"], device=self.device, dtype=torch.long, requires_grad=False)]
        
    def get_reconstructed_projected_gravity(self):
        #return: Dim(num_envs,prediction_horizon,3)
        return self.obs_pred[:,:, torch.tensor(self.target_fld_state_state_idx_dict["projected_gravity"], device=self.device, dtype=torch.long, requires_grad=False)]

    def get_reconstructed_dof_pos_leg_fl(self):
        #return: Dim(num_envs,prediction_horizon,3)
        return self.obs_pred[:,:, torch.tensor(self.target_fld_state_state_idx_dict["dof_pos_leg_fl"], device=self.device, dtype=torch.long, requires_grad=False)]

    def get_reconstructed_dof_pos_leg_hl(self):
        #return: Dim(num_envs,prediction_horizon,3)
        return self.obs_pred[:,:, torch.tensor(self.target_fld_state_state_idx_dict["dof_pos_leg_hl"], device=self.device, dtype=torch.long, requires_grad=False)]

    def get_reconstructed_dof_pos_leg_fr(self):
        #return: Dim(num_envs,prediction_horizon,3)
        return self.obs_pred[:,:, torch.tensor(self.target_fld_state_state_idx_dict["dof_pos_leg_fr"], device=self.device, dtype=torch.long, requires_grad=False)]

    def get_reconstructed_dof_pos_leg_hr(self):
        #return: Dim(num_envs,prediction_horizon,3)
        return self.obs_pred[:,:, torch.tensor(self.target_fld_state_state_idx_dict["dof_pos_leg_hr"], device=self.device, dtype=torch.long, requires_grad=False)]

    # def get_reconstructed_dof_feet_pos_fl(self):
    #     return self.obs_pred[:,:, torch.tensor(self.target_fld_state_state_idx_dict["feet_pos_fl"], device=self.device, dtype=torch.long, requires_grad=False)]

    # def get_reconstructed_dof_feet_pos_fr(self):
    #     return self.obs_pred[:,:, torch.tensor(self.target_fld_state_state_idx_dict["feet_pos_fr"], device=self.device, dtype=torch.long, requires_grad=False)]

    # def get_reconstructed_dof_feet_pos_hl(self):
    #     return self.obs_pred[:,:, torch.tensor(self.target_fld_state_state_idx_dict["feet_pos_hl"], device=self.device, dtype=torch.long, requires_grad=False)]

    # def get_reconstructed_dof_feet_pos_hr(self):
    #     return self.obs_pred[:,:, torch.tensor(self.target_fld_state_state_idx_dict["feet_pos_hr"], device=self.device, dtype=torch.long, requires_grad=False)]

    def get_reconstructed_dof_pos(self):
        #return: Dim(num_envs,prediction_horizon,12)
        return torch.stack(
            (
                self.get_reconstructed_dof_pos_leg_fl(),
                self.get_reconstructed_dof_pos_leg_hl(),
                self.get_reconstructed_dof_pos_leg_fr(),
                self.get_reconstructed_dof_pos_leg_hr(),
                ),dim=2
            ).reshape(self.num_envs,self.fld_observation_horizon,-1)

if __name__ == "__main__":
    pass