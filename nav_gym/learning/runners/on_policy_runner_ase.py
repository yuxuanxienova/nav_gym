#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# python
import os
import time
import statistics
from collections import deque
from nav_gym.nav_legged_gym.envs.local_nav_env import LocalNavEnv
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

# torch
import torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd
# rsl-rl
from nav_gym.learning.algorithms.ppo_ase import PPO_ASE
from nav_gym.learning.modules.actor_critic import ActorCritic, ActorCriticSeparate
from nav_gym.learning.modules.privileged_training.teacher_models import TeacherModelBase
from nav_gym.learning.modules.normalizer_module import EmpiricalNormalization
from nav_gym.learning.env import VecEnv
from nav_gym.learning.utils import store_code_state
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.learning.distribution.gaussian import Gaussian
from nav_gym.learning.distribution.beta_distribution import BetaDistribution
from nav_gym.learning.modules.navigation.local_nav_model import NavPolicyWithMemory
from nav_gym.nav_legged_gym.test.interactive_module import InteractModuleVelocity
import numpy as np
def load_model(obs_names_list, arch_cfg, obs_dict, num_actions, empirical_normalization):
    # Define observation space
    obs_shape_dict = {}
    obs_dim = 0
    for name in obs_names_list:
        obs_shape = obs_dict[name].shape
        obs_shape_single = obs_shape[1:]
        dim = obs_shape_single.numel()

        obs_dim += dim
        # save name and dimension
        obs_shape_dict[name] = obs_shape_single

    # Define model
    model_cls = eval(arch_cfg["model_class"])
    return model_cls(obs_shape_dict, num_actions, arch_cfg, empirical_normalization=empirical_normalization)


# JL: THIS DOES NOT SUPPORT ASYMMETRIC AC MODELS
# JL: THIS ONLY WORKS WITH TEACHER_STUDENT SETUP IN WHEELED_LEGGED_ENV
class OnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.actor_critic_cfg = train_cfg["actor_critic"]
        self.action_dist_cfg = self.actor_critic_cfg["action_distribution"]
        self.device = device
        self.env:LocalNavEnv = env
        self.num_envs = self.env.num_envs
        self.empirical_normalization = self.cfg["empirical_normalization"]

        obs, extras = self.env.get_observations()

        # Define observation space
        # obs_names = self.env.cfg.observations.teacher_obs_list
        obs_names = class_to_dict(self.env.cfg.observations).keys()

        # Define actor critic model
        if "num_logits" in self.action_dist_cfg:
            num_logits = self.action_dist_cfg["num_logits"]
        else:
            num_logits = self.env.num_actions

        actor_model = load_model(
            obs_names,
            self.actor_critic_cfg["actor_architecture"],
            extras["observations"],
            num_logits,
            self.empirical_normalization,
        )
        critic_model = load_model(
            obs_names,
            self.actor_critic_cfg["critic_architecture"],
            extras["observations"],
            1,
            self.empirical_normalization,
        )

        # Define action distribution
        action_dist_class = eval(self.action_dist_cfg["model_class"])
        action_dist = action_dist_class(self.env.num_actions, self.action_dist_cfg)

        # Define actor critic model
        actor_critic_class = eval(self.actor_critic_cfg["class_name"])  # ActorCritic
        actor_critic = actor_critic_class(actor_model, critic_model, action_dist).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO_ASE = alg_class(actor_critic, 
                                      device=self.device, 
                                      num_envs=self.num_envs,
                                      num_transitions_per_env=self.num_steps_per_env ,
                                      **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [actor_model.num_obs],
            [critic_model.num_obs],
            [self.env.num_actions],
        )
        self.asymmetric_critic = False
        self.critic_obs_list = None
        if actor_model.num_obs != critic_model.num_obs:
            self.asymmetric_critic = True
            self.critic_obs_list = critic_model.obs_names

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        # self.git_status_repos = [nav_gym.learning.__file__]

        #ASE
        self._latent_reset_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._latent_steps_min = 1
        self._latent_steps_max = 150
        self.runner_step_count = 0

        self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.device)
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        self.env.reset()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        if "critic" in extras["observations"]:
            critic_obs = extras["observations"]["critic"]
        else:
            critic_obs = obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    #--ASE--
                    self.runner_step_count += 1
                    self._update_latents()
                    #-------
                    actions = self.alg.act(obs, critic_obs)
                    log_prob = self.alg.get_log_prob(actions)
                    self.alg.store_transition(actions, log_prob, obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    #--ASE--
                    amp_obs = infos["observations"]["amp_obs"]
                    self.alg.amp_obs_storage.add_amp_obs_to_buffer(infos["observations"]["amp_obs"], i)
                    disc_r, enc_r = self._calc_amp_rewards(amp_obs, self.alg.ase_latents)
                    rewards = rewards + disc_r + enc_r
                    #-------
                    self.alg.process_env_step(rewards, dones, infos)
                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        mean_reward_episode = cur_reward_sum[new_ids]/cur_episode_length[new_ids]
                        rewbuffer.extend(mean_reward_episode[:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                #--ASE--
                self.alg.amp_obs_storage.add_transition_to_data()
                self.alg.amp_obs_storage.clear_buffer()
                #-------

            mean_value_loss, mean_surrogate_loss, mean_entropy_bonus = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
                ep_infos.clear()

        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))
        # TODO(areske): return something less noisy, e.g. exponential moving average
        return statistics.mean(rewbuffer)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        actor_log_data_dict = self.alg.actor_critic.log_info()

        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/entropy", locs["mean_entropy_bonus"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        for key, value in actor_log_data_dict.items():
            self.writer.add_scalar("Actor/" + key, value, locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Entropy bonus:':>{pad}} {locs['mean_entropy_bonus']:.4f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Entropy bonus:':>{pad}} {locs['mean_entropy_bonus']:.4f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )

        print(log_string)
    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)
    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        print(f"Loaded model from {path} at iteration {self.current_learning_iteration}")
        return loaded_dict["infos"]
    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        return policy
    def train_mode(self):
        self.alg.actor_critic.train()
    def eval_mode(self):
        self.alg.actor_critic.eval()
    def _update_latents(self):
        new_latent_envs = self._latent_reset_steps <= self.runner_step_count
        need_update = torch.any(new_latent_envs)
        if (need_update):
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            n = len(new_latent_env_ids)
            z = self.alg.sample_latents(n)
            self.alg.set_ase_latents(z,new_latent_env_ids)
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(self._latent_reset_steps[new_latent_env_ids],
                                                                               low=self._latent_steps_min, 
                                                                               high=self._latent_steps_max)
        return
    def _calc_amp_rewards(self, amp_obs, ase_latents):
        proc_amp_obs = self._amp_input_mean_std(amp_obs)
        with torch.no_grad():
            disc_logits,mu_q = self.alg.discriminator_encoder(proc_amp_obs)
        disc_r = self._calc_disc_rewards(disc_logits)
        enc_r = self._calc_enc_rewards(mu_q, ase_latents)
        return disc_r, enc_r
    def _calc_disc_rewards(self, disc_logits):
        prob = 1 / (1 + torch.exp(-disc_logits)) 
        disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
        return disc_r
    def _calc_enc_rewards(self, mu_q, ase_latents):
        enc_r = torch.clamp_min(torch.sum(mu_q * ase_latents, dim=-1, keepdim=True), 0.0).to(self.device)
        return enc_r

    



w



    # def _update_amp_demos(self):
    #     new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
    #     self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
    #     return
    # def _fetch_amp_obs_demo(self, num_samples):
    #     amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
    #     return amp_obs_demo
    
    # def fetch_amp_obs_demo(self, num_samples):

    #     if (self._amp_obs_demo_buf is None):
    #         self._build_amp_obs_demo_buf(num_samples)
    #     else:
    #         assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
    #     motion_ids = self._motion_lib.sample_motions(num_samples)
        
    #     # since negative times are added to these values in build_amp_obs_demo,
    #     # we shift them into the range [0 + truncate_time, end of clip]
    #     truncate_time = self.dt * (self._num_amp_obs_steps - 1)
    #     motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
    #     motion_times0 += truncate_time

    #     amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
    #     self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
    #     amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

    #     return amp_obs_demo_flat
    # def build_amp_obs_demo(self, motion_ids, motion_times0):
    #     dt = self.dt

    #     motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
    #     motion_times = motion_times0.unsqueeze(-1)
    #     time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
    #     motion_times = motion_times + time_steps

    #     motion_ids = motion_ids.view(-1)
    #     motion_times = motion_times.view(-1)
    #     root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
    #            = self._motion_lib.get_motion_state(motion_ids, motion_times)
    #     amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
    #                                           dof_pos, dof_vel, key_pos,
    #                                           self._local_root_obs, self._root_height_obs,
    #                                           self._dof_obs_size, self._dof_offsets)
    #     return amp_obs_demo
    # def get_motion_state(self, motion_ids, motion_times):
    #     n = len(motion_ids)
    #     num_bodies = self._get_num_bodies()
    #     num_key_bodies = self._key_body_ids.shape[0]

    #     motion_len = self._motion_lengths[motion_ids]
    #     num_frames = self._motion_num_frames[motion_ids]
    #     dt = self._motion_dt[motion_ids]

    #     frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

    #     f0l = frame_idx0 + self.length_starts[motion_ids]
    #     f1l = frame_idx1 + self.length_starts[motion_ids]

    #     root_pos0 = self.gts[f0l, 0]
    #     root_pos1 = self.gts[f1l, 0]

    #     root_rot0 = self.grs[f0l, 0]
    #     root_rot1 = self.grs[f1l, 0]

    #     local_rot0 = self.lrs[f0l]
    #     local_rot1 = self.lrs[f1l]

    #     root_vel = self.grvs[f0l]

    #     root_ang_vel = self.gravs[f0l]
        
    #     key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
    #     key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

    #     dof_vel = self.dvs[f0l]

    #     vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0, key_pos1]
    #     for v in vals:
    #         assert v.dtype != torch.float64


    #     blend = blend.unsqueeze(-1)

    #     root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

    #     root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

    #     blend_exp = blend.unsqueeze(-1)
    #     key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
    #     local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
    #     dof_pos = self._local_rotation_to_dof(local_rot)

    #     return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos
    
    # def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
    #                         local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    #     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    #     root_h = root_pos[:, 2:3]
    #     heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    #     if (local_root_obs):
    #         root_rot_obs = quat_mul(heading_rot, root_rot)
    #     else:
    #         root_rot_obs = root_rot
    #     root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
        
    #     if (not root_height_obs):
    #         root_h_obs = torch.zeros_like(root_h)
    #     else:
    #         root_h_obs = root_h
        
    #     local_root_vel = quat_rotate(heading_rot, root_vel)
    #     local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    #     root_pos_expand = root_pos.unsqueeze(-2)
    #     local_key_body_pos = key_body_pos - root_pos_expand
        
    #     heading_rot_expand = heading_rot.unsqueeze(-2)
    #     heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    #     flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    #     flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
    #                                             heading_rot_expand.shape[2])
    #     local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    #     flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
        
    #     dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    #     obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    #     return obs