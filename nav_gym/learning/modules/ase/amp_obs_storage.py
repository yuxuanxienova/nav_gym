
# import os 
# from nav_gym import NAV_GYM_ROOT_DIR
# import random
# import torch
# class AMPObsStorage:
#     def __init__(self,obs_dim,num_envs,num_transitons_per_env, trajectory_capacity):
#         #parse args
#         self.num_envs = num_envs
#         self.capacity = trajectory_capacity
#         self.obs_dim = obs_dim
#         self.num_transitons_per_env = num_transitons_per_env

#         #initialize buffers
#         self.transition_buffer = torch.zeros(self.num_envs,self.num_transitons_per_env,self.obs_dim)
#         self.data = torch.zeros(self.capacity,self.num_envs,self.num_transitons_per_env,self.obs_dim)

#         self.length = 0

#         self.save_root = os.path.join(NAV_GYM_ROOT_DIR + "/resources/anymal_d/datasets/record/")
#         self.save_name = 'sampled_amp_observations.pt'
#         self.save_path = os.path.join(self.save_root, self.save_name)

#         self.current_transition_idx = 0
#     def add_amp_obs_to_buffer(self, amp_obs,transition_idx):
#         #amp_obs: [num_envs,obs_dim]
#         self.transition_buffer[:,transition_idx,:] = amp_obs
#         self.current_transition_idx = transition_idx

#     def add_transition_to_data(self):
#         if self.length < self.capacity:
#             # Buffer not full, place the new transition at the next available position
#             self.data[self.length] = self.transition_buffer
#             self.length += 1
#         else:
#             # Buffer is full, shift data to the left to discard the oldest transition
#             self.data[:-1] = self.data[1:]
#             self.data[-1] = self.transition_buffer
#         print(f"[INFO] Length: {self.length}")

#     def is_ready(self,sample_length):
#         return self.length > sample_length
#     def clear_buffer(self):
#         self.transition_buffer = torch.zeros(self.num_envs,self.num_transitons_per_env,self.obs_dim)
#     # def is_ready(self):
#     #     return 
#     def get_amp_traj_shape(self):
#         return self.data[0].shape
#     # def get_current_obs(self,sample_length):
#     #     #----Error Handling----
#     #     if self.num_transitons_per_env < sample_length:
#     #         print("[ERROR]Sample length is greater than the number of transitions per environment")
#     #         exit(1)
#     #     #--------------------
#     #     #self.data[self.count]: [num_envs,num_transitons_per_env,obs_dim]
#     #     obs_amp = self.data[self.count-1][:,:sample_length,:]
#     #     #obs_amp: [num_envs,sample_length,obs_dim]
#     #     return obs_amp
#     def get_current_obs(self, sample_length):
#         # ---- Error Handling ----
#         total_transitions = self.length * self.num_transitons_per_env + self.transition_buffer.shape[1]
#         if total_transitions < sample_length:
#             print("[ERROR] Not enough transitions to sample the requested sample_length")
#             exit(1)
#         # --------------------

#         # Collect all transitions from data and transition_buffer
#         # data: [length, num_envs, num_transitons_per_env, obs_dim]
#         # Only use the stored data up to self.length
#         data = self.data[:self.length]  # Shape: [count, num_envs, num_transitons_per_env, obs_dim]

#         # Reshape data to [num_envs, length * num_transitons_per_env, obs_dim]
#         data = data.permute(1, 0, 2, 3).contiguous()  # [num_envs, count, num_transitons_per_env, obs_dim]
#         data = data.view(self.num_envs, -1, self.obs_dim)  # [num_envs, count * num_transitons_per_env, obs_dim]

#         # Collect transitions from transition_buffer
#         buffer_transitions = self.transition_buffer[:,:self.current_transition_idx,:]  # [num_envs, num_transitons_per_env, obs_dim]

#         # Concatenate data and transition_buffer along the time dimension
#         if self.current_transition_idx == self.num_transitons_per_env-1:
#             all_transitions = data
#         else:
#             all_transitions = torch.cat((data, buffer_transitions), dim=1)  # [num_envs, total_transitions, obs_dim]

#         # Now get the last sample_length transitions
#         obs_amp = all_transitions[:, -sample_length:, :]  # [num_envs, sample_length, obs_dim]

#         return obs_amp
#     def sample_amp_obs_batch(self, sample_length, num_sample=1):
#         #----Error Handling----
#         if self.num_transitons_per_env < sample_length:
#             print("[ERROR]Sample length is greater than the number of transitions per environment")
#             exit(1)
#         #--------------------    
#         indices = torch.randint(0, self.length, (num_sample,))
#         #self.data[indices]: [num_sample,num_envs,num_transitons_per_env,obs_dim]
#         obs_amp_replay = self.data[indices][:,:,:sample_length,:].flatten(0,1)  
#         #obs_amp_replay: [num_sample*num_envs,sample_length,obs_dim]

#         #save
#         # torch.save(obs_amp_replay, self.save_path)
#         return obs_amp_replay

import os
import random
import torch
from collections import deque
from nav_gym import NAV_GYM_ROOT_DIR

class AMPObsStorage:
    def __init__(self, obs_dim, num_envs, num_transitions_per_env, trajectory_capacity):
        # Parse arguments
        self.num_envs = num_envs
        self.capacity = trajectory_capacity
        self.obs_dim = obs_dim
        self.num_transitions_per_env = num_transitions_per_env

        # Initialize buffers
        self.transition_buffer = torch.zeros(self.num_envs, self.num_transitions_per_env, self.obs_dim)
        self.data = deque(maxlen=self.capacity)

        self.save_root = os.path.join(NAV_GYM_ROOT_DIR, "resources/anymal_d/datasets/record/")
        self.save_name = 'sampled_amp_observations.pt'
        self.save_path = os.path.join(self.save_root, self.save_name)

        self.current_transition_idx = 0

    def add_amp_obs_to_buffer(self, amp_obs, transition_idx):
        # amp_obs: [num_envs, obs_dim]
        self.transition_buffer[:, transition_idx, :] = amp_obs
        self.current_transition_idx = transition_idx

    def add_transition_to_data(self):
        # Append the current transition buffer to the deque
        self.data.append(self.transition_buffer.clone())

        # Reset the transition buffer and index for the next set of transitions
        self.transition_buffer = torch.zeros(self.num_envs, self.num_transitions_per_env, self.obs_dim)
        self.current_transition_idx = 0

        print(f"[INFO] Length: {len(self.data)}")

    def is_ready(self, sample_length):
        total_transitions = len(self.data) 
        return total_transitions > sample_length

    def clear_buffer(self):
        self.transition_buffer = torch.zeros(self.num_envs, self.num_transitions_per_env, self.obs_dim)
        self.current_transition_idx = 0

    def get_amp_traj_shape(self):
        if len(self.data) > 0:
            return self.data[0].shape
        else:
            return None

    def get_current_obs(self, sample_length):
        # ---- Error Handling ----
        total_transitions = len(self.data) * self.num_transitions_per_env + (self.current_transition_idx + 1)
        if total_transitions < sample_length:
            print("[ERROR] Not enough transitions to sample the requested sample_length")
            exit(1)
        # --------------------

        # Collect all transitions from data and transition_buffer
        if len(self.data) > 0:
            data_list = list(self.data)
            # Concatenate along the time dimension
            data = torch.cat(data_list, dim=1)  # [num_envs, total_transitions_in_data, obs_dim]
        else:
            data = torch.empty(self.num_envs, 0, self.obs_dim)

        # Include the current transition buffer up to the current index
        buffer_transitions = self.transition_buffer[:, :self.current_transition_idx + 1, :]
        # Concatenate data and buffer_transitions
        if self.current_transition_idx == 0:
            all_transitions = data
        else:
            all_transitions = torch.cat((data, buffer_transitions), dim=1)  # [num_envs, total_transitions, obs_dim]

        # Get the last sample_length transitions
        obs_amp = all_transitions[:, -sample_length:, :]  # [num_envs, sample_length, obs_dim]
        #save
        # torch.save(obs_amp, self.save_root + "sampled_amp_observations{0}.pt".format(random.randint(0, 1000)))
        return obs_amp

    def sample_amp_obs_batch(self, sample_length, num_sample=1):
        # ---- Error Handling ----
        if self.num_transitions_per_env < sample_length:
            print("[ERROR] Sample length is greater than the number of transitions per environment")
            exit(1)
        if len(self.data) < num_sample:
            print("[ERROR] Not enough samples to draw the requested number of samples")
            exit(1)
        # --------------------

        # Randomly sample indices without replacement
        indices = random.sample(range(len(self.data)), num_sample)
        # Collect the sampled data
        sampled_tensors = [self.data[idx][:, :sample_length, :] for idx in indices]  # List of [num_envs, sample_length, obs_dim]

        # Stack sampled tensors into one tensor
        obs_amp_replay = torch.stack(sampled_tensors, dim=0)  # [num_sample, num_envs, sample_length, obs_dim]

        # Flatten the first two dimensions (num_sample and num_envs)
        obs_amp_replay = obs_amp_replay.view(-1, sample_length, self.obs_dim)  # [num_sample * num_envs, sample_length, obs_dim]

        return obs_amp_replay