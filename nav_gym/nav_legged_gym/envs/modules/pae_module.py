from isaacgym import gymutil, gymapi
from isaacgym.torch_utils import (
    quat_rotate,
    quat_rotate_inverse,
    get_euler_xyz,
    quat_from_euler_xyz,
)
import numpy as np
import torch
from nav_gym.learning.modules.samplers.offline import OfflineSamplerPAE
from nav_gym.learning.modules.samplers.gmm import GMMSampler
from nav_gym import NAV_GYM_ROOT_DIR
from nav_gym.learning.modules.fld.fld import FLD
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.envs.locomotion_pae_env import LocomotionPAEEnv
    ANY_ENV = Union[LocomotionPAEEnv]
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

class FLD_PAEModule:
    def __init__(self,env:"ANY_ENV"):
        #1. Parse environment arguments
        self.env_cfg = env.cfg
        self.fld_cfg = env.cfg.fld
        self.task_sampler_cfg=env.cfg.task_sampler
        self.fld_latent_channel = self.fld_cfg.latent_channel
        self.fld_observation_horizon = self.fld_cfg.observation_horizon
        self.full_state_idx_dict = self.fld_cfg.state_idx_dict

        self.robot = env.robot
        self.base_pos = self.robot.root_pos_w
        self.base_quat = self.robot.root_quat_w
        self.base_lin_vel = self.robot.root_lin_vel_b#TODO check if this is correct .root_lin_vel_b??!!!
        self.base_ang_vel = self.robot.root_ang_vel_b
        self.projected_gravity = self.robot.projected_gravity_b
        self.dof_pos = self.robot.dof_pos
        self.default_dof_pos = self.robot.default_dof_pos
        self.dof_vel = self.robot.dof_vel

        self.terrain = env.terrain

        self.num_envs = env.num_envs
        self.device = env.device


        self.dt = env.dt
        self.max_episode_length_s = self.env_cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.env_draw_handle = env.envs[0]
        self.gym = env.gym
        self.viewer = env.viewer
        self.pae_module_step_counter = env.common_step_counter

        self.reward_manager = env.reward_manager
        self.tracking_reconstructed_terms = []
        for rew_name,rew_fun in self.reward_manager.reward_functions.items():
            if "reward_tracking_reconstructed" in rew_name:
                self.tracking_reconstructed_terms.append(rew_name)

        #2. other arguments
        self.target_fld_state_state_idx_dict = {}
        current_length = 0
        for state, ids in self.full_state_idx_dict.items():
            if (state != "base_pos") and (state != "base_quat"):
                self.target_fld_state_state_idx_dict[state] = list(range(current_length, current_length + len(ids)))
                current_length = current_length + len(ids)

        # Initialize data for plotting DOF positions
        self.dof_pos_robot_history = []
        self.dof_pos_motion_history = []
        self.time_steps = []

        # Initialize plotting
        self.plot_initialized = False
        self.num_dofs = self.dof_pos.shape[1]  # Assuming self.dof_pos shape is (num_envs, num_dofs)

        #3. Initialize buffers
        self.pae_module_step_counter = 0
        self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.cur_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.fld_dim_of_interest = torch.cat([torch.tensor(ids, device=self.device, dtype=torch.long, requires_grad=False) for state, ids in self.full_state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
        self.fld_observation_dim = len(self.fld_dim_of_interest)
        # self.fld_observation_buf = torch.zeros(self.num_envs, self.fld_observation_horizon, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.fld_state = torch.zeros(self.num_envs, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.latent_encoding = torch.zeros(self.num_envs, self.fld_cfg.latent_channel, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.params = torch.zeros(self.num_envs, self.fld_cfg.latent_channel, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.latent_manifold = torch.zeros(self.num_envs, self.fld_cfg.latent_channel * 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_fld_state = torch.zeros(self.num_envs, self.fld_observation_dim, dtype=torch.float, device=self.device, requires_grad=False)
        #4. Initialize Task Sampler
        if self.task_sampler_cfg.name == "OfflineSamplerPAE":
            self.task_sampler =  OfflineSamplerPAE(self.device)
            self.task_sampler.load_data(self.fld_cfg.load_root_pretrain+"/latent_params_pae.pt")
            self.task_sampler.load_motions(self.fld_cfg.load_root_pretrain+"/state_transition_pae.pt")
        else:
            print("[ERROR] Task Sampler not implemented")
            raise NotImplementedError
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
        print("[INFO][self.cur_steps[0]:{0}] fld module on_env_post_physics_step".format(self.cur_steps[0]))
        self.pae_module_step_counter += 1
        self._update_module_buffers()
        self._update_fld_observation_buf()
        self._update_latent_phase_pae()

        robot_root_lin_vel_w = quat_rotate(self.base_quat,self.fld_state[:,self.target_fld_state_state_idx_dict["base_lin_vel"]])
        target_root_lin_vel_w = quat_rotate(self.base_quat,self.target_fld_state[:,self.target_fld_state_state_idx_dict["base_lin_vel"]])
        
        self.visualize_velocities(robot_root_lin_vel_w, target_root_lin_vel_w)
        # Update the plot every N steps to reduce overhead
        self._update_plot_data()
        N = 10  # Update plot every N steps
        if self.pae_module_step_counter % N == 0:
            self._update_plot()

    def _update_module_buffers(self):
        self.base_pos = self.robot.root_pos_w
        self.base_quat = self.robot.root_quat_w
        self.base_lin_vel = self.robot.root_lin_vel_b#TODO check if this is correct .root_lin_vel_b??!!!
        self.base_ang_vel = self.robot.root_ang_vel_b
        self.projected_gravity = self.robot.projected_gravity_b
        self.dof_pos = self.robot.dof_pos
        self.default_dof_pos = self.robot.default_dof_pos
        self.dof_vel = self.robot.dof_vel

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
            self.fld_state[:, value] = full_state[:, self.full_state_idx_dict[key]].clone()
        # self.fld_observation_buf[:, :-1] = self.fld_observation_buf[:, 1:].clone()
        # self.fld_observation_buf[:, -1] = self.fld_state.clone()
    def _update_latent_phase_pae(self):
        #1. get current latent encoding
        #self.task_sampler.data: Dim: [n_motions, n_steps, n_latent_features]=[10, 169, 16]
        self.num_steps = self.task_sampler.data.shape[1]
        # print("[Debug][pae_module][_update_latent_phase_pae]Disable self.cur_steps = (self.cur_steps + 1) % self.num_steps")
        self.cur_steps = (self.cur_steps + 1) % self.num_steps
        self.latent_encoding_target = self.task_sampler.data[self.motion_idx, self.cur_steps, :].view(self.num_envs, 4, self.fld_latent_channel).swapaxes(1, 2)
        #self.latent_encoding_target: Dim(num_envs,latent_channel,4)

        #2. update latent encoding
        self.latent_encoding = self.latent_encoding_target.clone()
        #self.latent_encoding: Dim(num_envs,latent_channel,4)
        phase_t = self.latent_encoding[:, :, 0]#Dim(num_envs,latent_channel)
        frequency_t = self.latent_encoding[:, :, 1]#Dim(num_envs,latent_channel)
        amplitude_t = self.latent_encoding[:, :, 2]#Dim(num_envs,latent_channel)
        offset_t = self.latent_encoding[:, :, 3]#Dim(num_envs,latent_channel)

        #3. reconstruct z
        reconstructed_z = amplitude_t.unsqueeze(-1) * torch.sin(2 * torch.pi * (frequency_t.unsqueeze(-1) * self.fld.args + phase_t.unsqueeze(-1))) + offset_t.unsqueeze(-1)
        
        #4. observation from dataset
        #self.task_sampler.motions: Dim: [n_motions*n_trajs, n_windows, n_obs_dim, obs_horizon]=[10, 219, 21, 31]
        target_fld_state_wins_raw = self.task_sampler.motions[self.motion_idx, self.cur_steps,].swapaxes(1, 2)
        #target_fld_state_wins_raw: Dim: [num_envs,obs_horizon,obs_dim]
        self.target_fld_state[:] = target_fld_state_wins_raw[:, -1, :] * self.state_transitions_std + self.state_transitions_mean#Dim(num_envs,input_channel)
        #self.target_fld_state: Dim(num_envs,obs_dim)

    def on_env_reset_idx(self,env_ids):
        if self.env_cfg.task_sampler.curriculum and self.pae_module_step_counter % self.max_episode_length == 0:
            self.update_task_sampler_curriculum(env_ids)
        self._sample_latent_encoding(env_ids)
    def on_env_resample_commands(self,env_ids):
        self._sample_latent_encoding(env_ids)
    def update_task_sampler_curriculum(self, env_ids):
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if not hasattr(self, "task_sampler_curriculum_flag"):
            self.task_sampler_curriculum_flag = torch.zeros((self.num_envs, self.task_sampler.motions.shape[0]), dtype=torch.long, device=self.device, requires_grad=False)
        self.task_sampler_curriculum_flag[env_ids, self.motion_idx[env_ids]] = True
        for name in self.tracking_reconstructed_terms:
            self.task_sampler_curriculum_flag[env_ids, self.motion_idx[env_ids]] &= (torch.mean(self.reward_manager.episode_sums[name][env_ids]) / self.max_episode_length
            > self.env_cfg.task_sampler.curriculum_performance_threshold * self.reward_manager.reward_params[name]["scale"])
    def _sample_latent_encoding(self,env_ids):
        if len(env_ids) == 0:
            return
        if self.env_cfg.task_sampler.curriculum:
            self.motion_idx[env_ids], self.cur_steps[env_ids] = self.task_sampler.sample_curriculum(len(env_ids), self.task_sampler_curriculum_flag)
        else:
            self.motion_idx[env_ids], self.cur_steps[env_ids] = self.task_sampler.sample(len(env_ids))
        # self.motion_idx[env_ids], self.cur_steps[env_ids] = self.task_sampler.sample(len(env_ids))
        #-----------debug-use------------
        print("[Debug][pae_module][_sample_latent_encoding] Setting motion_idx and cur_steps to 0")
        self.motion_idx[env_ids] = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.cur_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        #--------------------------------
        #self.task_sampler.data: Dim: [n_motions, n_steps, n_latent_features]=[10, 169, 16]
        #self.task_sampler.data[self.motion_idx[env_ids], self.cur_steps[env_ids]]: Dim: [num_envs, n_latent_features]=[num_envs, 16]
        self.latent_encoding[env_ids, :, :] = self.task_sampler.data[self.motion_idx[env_ids], self.cur_steps[env_ids]].view(len(env_ids), 4, self.fld_latent_channel).swapaxes(1, 2)
        #self.latent_encoding[env_ids, :, :] : Dim: [len(env_ids), 4, n_latent_features]=[num_envs, 4, 4]
        
        if self.fld_cfg.filter_latent:
            if not hasattr(self.task_sampler, "filtered_data"):
                self.task_sampler.load_filtered_encoding(n_unfolds=self.fld_cfg.filter_size)
                #filtered_data: Dim: [n_motions, n_steps, window_len]=[10, 170, 5]
            self.latent_encoding[env_ids, :, 1:] = self.task_sampler.filtered_data[self.motion_idx[env_ids], self.cur_steps[env_ids], self.fld_latent_channel:].view(len(env_ids), 3, self.fld_latent_channel).swapaxes(1, 2)
        
            
    
    # Visualization method for velocities
    def visualize_velocities(self, robot_root_lin_vel, target_root_lin_vel):
        # Loop through each environment to draw velocities
 
        env_handle = self.env_draw_handle
        i=0
        self.velocity_scale=1

        #offset
        offset = np.array([0,0,1])
        # Get the base position of the robot
        base_pos = self.robot.root_pos_w[i].cpu().numpy() + offset

        # Robot's current velocity
        robot_vel = robot_root_lin_vel[i].cpu().numpy() * self.velocity_scale
        robot_vel_end = base_pos + robot_vel

        # Target velocity
        target_vel = target_root_lin_vel[i].cpu().numpy() * self.velocity_scale
        target_vel_end = base_pos + target_vel

        # Draw the robot's velocity vector in blue
        p1 = gymapi.Vec3(*base_pos)
        p2 = gymapi.Vec3(*robot_vel_end)
        color_blue = gymapi.Vec3(0, 0, 1)
        gymutil.draw_line(p1, p2, color_blue, self.gym, self.viewer, self.env_draw_handle)

        # Draw the target velocity vector in red
        p3 = gymapi.Vec3(*base_pos)
        p4 = gymapi.Vec3(*target_vel_end)
        color_red = gymapi.Vec3(1, 0, 0)
        gymutil.draw_line(p3, p4, color_red, self.gym, self.viewer, self.env_draw_handle)
    # Method to initialize the plot
    def _initialize_plot(self):
        plt.ion()  # Turn on interactive mode
        self.fig, self.axs = plt.subplots(self.num_dofs, 1, figsize=(10, 2 * self.num_dofs), sharex=True)
        if self.num_dofs == 1:
            self.axs = [self.axs]
        self.lines_robot = []
        self.lines_desired = []
        for i in range(self.num_dofs):
            line_robot, = self.axs[i].plot([], [], label='Robot DOF {}'.format(i))
            line_desired, = self.axs[i].plot([], [], label='Desired DOF {}'.format(i))
            self.axs[i].set_ylabel('DOF Position')
            self.axs[i].legend()
            self.lines_robot.append(line_robot)
            self.lines_desired.append(line_desired)
        self.axs[-1].set_xlabel('Time Step')
        self.plot_initialized = True
    def _update_plot_data(self):
        # Collect data for plotting
        robot_dof_pos = self.dof_pos.clone()
        desired_dof_pos = self.get_target_dof_pos()

        # Select the first environment for plotting
        robot_dof_pos_env0 = robot_dof_pos[0].cpu().numpy()
        desired_dof_pos_env0 = desired_dof_pos[0].cpu().numpy()

        self.dof_pos_robot_history.append(robot_dof_pos_env0)
        self.dof_pos_motion_history.append(desired_dof_pos_env0)
        self.time_steps.append(self.pae_module_step_counter)


    # Method to update the plot
    def _update_plot(self):
        if not self.plot_initialized:
            self._initialize_plot()

        M = 100  # Number of time steps to display (adjust as needed)
        dof_pos_robot_history = np.array(self.dof_pos_robot_history)  # Shape: (num_time_steps, num_dofs)
        dof_pos_motion_history = np.array(self.dof_pos_motion_history)
        time_steps = np.array(self.time_steps)

        # Limit data to the last M points to keep the plot clean
        if len(time_steps) > M:
            dof_pos_robot_history = dof_pos_robot_history[-M:]
            dof_pos_motion_history = dof_pos_motion_history[-M:]
            time_steps = time_steps[-M:]
            # Adjust time steps to be sequential for plotting
            time_steps = np.arange(len(time_steps))

        for i in range(self.num_dofs):
            self.lines_robot[i].set_data(time_steps, dof_pos_robot_history[:, i])
            self.lines_desired[i].set_data(time_steps, dof_pos_motion_history[:, i])
            self.axs[i].relim()
            self.axs[i].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_target_dof_pos_leg_fl(self):
        #return: Dim(num_envs,3)
        return self.target_fld_state[:, torch.tensor(self.target_fld_state_state_idx_dict["dof_pos_leg_fl"], device=self.device, dtype=torch.long, requires_grad=False)]

    def get_target_dof_pos_leg_hl(self):
        #return: Dim(num_envs,3)
        return self.target_fld_state[:, torch.tensor(self.target_fld_state_state_idx_dict["dof_pos_leg_hl"], device=self.device, dtype=torch.long, requires_grad=False)]

    def get_target_dof_pos_leg_fr(self):
        #return: Dim(num_envs,3)
        return self.target_fld_state[:, torch.tensor(self.target_fld_state_state_idx_dict["dof_pos_leg_fr"], device=self.device, dtype=torch.long, requires_grad=False)]

    def get_target_dof_pos_leg_hr(self):
        #return: Dim(num_envs,3)
        return self.target_fld_state[:, torch.tensor(self.target_fld_state_state_idx_dict["dof_pos_leg_hr"], device=self.device, dtype=torch.long, requires_grad=False)]

    def get_target_dof_pos(self):
        #return: Dim(num_envs,12)
        return torch.stack(
            (
                self.get_target_dof_pos_leg_fl(),
                self.get_target_dof_pos_leg_hl(),
                self.get_target_dof_pos_leg_fr(),
                self.get_target_dof_pos_leg_hr(),
                ),dim=2
            ).reshape(self.num_envs,-1)
if __name__ == "__main__":
    pass