# isaac-gym
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_from_euler_xyz, quat_apply
# python
from copy import deepcopy
import torch
import numpy as np
from typing import Tuple, Union, Dict, Any
import math
import torch
import abc
# legged-gym
from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
from nav_gym.nav_legged_gym.envs.config_locomotion_env import LocomotionEnvCfg
from nav_gym.nav_legged_gym.common.assets.robots.legged_robots.legged_robot import LeggedRobot
from nav_gym.nav_legged_gym.common.sensors.sensors import SensorBase, Raycaster
from nav_gym.nav_legged_gym.utils.math_utils import wrap_to_pi
from nav_gym.nav_legged_gym.common.terrain.terrain_unity import TerrainUnity
from nav_gym.nav_legged_gym.common.gym_interface import GymInterface
from nav_gym.nav_legged_gym.common.rewards.reward_manager import RewardManager
from nav_gym.nav_legged_gym.common.observations.observation_manager import ObsManager
from nav_gym.nav_legged_gym.common.terminations.termination_manager import TerminationManager
from nav_gym.nav_legged_gym.common.curriculum.curriculum_manager import CurriculumManager
from nav_gym.nav_legged_gym.common.sensors.sensor_manager import SensorManager
from nav_gym.nav_legged_gym.common.commands.command import CommandBase,UnifromVelocityCommand,UnifromVelocityCommandCfg,WaypointCommand,WaypointCommandCfg
from nav_gym.nav_legged_gym.utils.visualization_utils import BatchWireframeSphereGeometry
from nav_gym.nav_legged_gym.envs.config_local_nav_env import LocalNavEnvCfg
from nav_gym.nav_legged_gym.envs.modules.exp_memory import ExplicitMemory
from nav_gym.nav_legged_gym.envs.modules.pose_history import PoseHistoryData
import os
from nav_gym import NAV_GYM_ROOT_DIR
class LocalNavEnv:
    #-------- 1. Initialization -----------
    def __init__(self, cfg:LocalNavEnvCfg, ll_env_cls:LocomotionEnv) -> None:
        self.cfg = cfg
        #1. Parse the configuration
        cfg.ll_env_cfg.gym.headless = cfg.gym.headless
        cfg.ll_env_cfg.env.num_envs = cfg.env.num_envs
        #1.1 Create the low-level environment
        self.ll_env: LocomotionEnv = ll_env_cls(cfg.ll_env_cfg)
        self.ll_env.play_mode = True #enable play mode
        #1.2 Extract the necessary attributes
        self.sim = self.ll_env.sim
        self.gym = self.ll_env.gym
        self.gym_iface = self.ll_env.gym_iface
        self.num_envs = self.ll_env.num_envs
        self.device= self.ll_env.device
        self.robot = self.ll_env.robot
        self.terrain = self.ll_env.terrain
        self.viewer = self.ll_env.viewer
        self.num_actions = self.cfg.env.num_actions
        self.max_episode_length_s = self.cfg.env.episode_length_s # TODO 
        self.dt = self.ll_env.dt * self.cfg.hl_decimation
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        #1.3 Load the low-level policy
        scripted_model_path = os.path.join(NAV_GYM_ROOT_DIR, "resources/model/low_level/" )
        file_name = "ll_jit_model.pt"
        #Load the policy
        try:
            self.ll_policy = torch.jit.load(scripted_model_path + file_name).to("cuda:0")
            self.ll_policy.eval()
        except Exception as e:
            print("Loading scripted model failed:", e)
            exit(1)

        
        #2. Prepare mdp helper managers
        #2.1---Initialize the Local Navigation Modules---
        self.command_generator = WaypointCommand(WaypointCommandCfg(),env=self.ll_env)
        self.global_memory = ExplicitMemory(cfg.memory)
        self.wp_history = PoseHistoryData(cfg.memory)
        self.pose_history_exp = PoseHistoryData(cfg.memory)
        #-------------------------------------------------
        self._init_buffers()
        self.sensor_manager = SensorManager(self)
        self.reward_manager = RewardManager(self)
        self.obs_manager = ObsManager(self)
        self.termination_manager = TerminationManager(self)
        self.curriculum_manager = CurriculumManager(self)
        #3. Initialize the environment
        self.reset()
        self.obs_dict = self.obs_manager.compute_obs(self)
        #4. others
        self.num_obs = self.obs_manager.get_obs_dims_from_group("policy")
        self.num_privileged_obs = self.obs_manager.get_obs_dims_from_group("privileged")
        self.play_mode_ll = False
        self.play_mode_hl =True
    def _init_buffers(self):
        self.obs_dict = dict()
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.ll_reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = dict()

        self.pos_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_time_left = torch.zeros(self.num_envs, device=self.device)
        #Set the target position
        # self.pos_target[:, 0] = self.ll_env.terrain.x_goal
        # self.pos_target[:, 1] = self.ll_env.terrain.y_goal
        # self.pos_target[:, 2] = 0.5
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.command_generator.resample(env_ids)
        self.pos_target = self.command_generator.get_goal_position_command()

        # targets given to ll by hl x_vel, y_vel, yaw_vel
        self.command_x_vel = torch.zeros(self.num_envs, device=self.device)
        self.command_y_vel = torch.zeros(self.num_envs, device=self.device)
        self.command_yaw_vel = torch.zeros(self.num_envs, device=self.device)

        # metrics for losses
        self.total_sq_torques = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.total_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_pos = self.robot.root_pos_w.clone()

        #---Initialize the Local Navigation Module Buffers---
        self.first_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_error_z_check = torch.zeros(self.num_envs, device=self.device)
        self.previous_pos = self.robot.root_pos_w.clone()

        local_map_shape = [1]  # TODO: unused for now
        self.global_memory.init_buffers(self.num_envs, self.cfg.memory.max_node_cnt, local_map_shape, self.device)
        self.wp_history.init_buffers(self.num_envs, False, self.device)
        self.pose_history_exp.init_buffers(self.num_envs, False, self.device)
        # action-related buffers
        self.actions = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.scaled_action = torch.zeros(self.num_envs, 3, device=self.device)

        vel_cmd_scale = np.array(self.cfg.vel_cmd_scale)
        vel_cmd_offset = np.array(self.cfg.vel_cmd_offset)
        vel_cmd_max = np.array(self.cfg.vel_cmd_max)
        self.action_scale = torch.tensor(vel_cmd_scale, device=self.device, dtype=torch.float)
        self.action_offset = torch.tensor(vel_cmd_offset, device=self.device, dtype=torch.float)
        self.scaled_action_max = torch.tensor(vel_cmd_max, device=self.device, dtype=torch.float)

    #--------------2. Step ---------------
    def step(self, actions):
        self._preprocess_actions(actions)
        self.ll_reset_buf.zero_()
        for _ in range(self.cfg.hl_decimation):
            ll_actions = self._get_ll_actions()
            _,_, _, dones, _ = self.ll_env.step(ll_actions)
            self.ll_reset_buf |= dones
            #visualization
            self._draw_hl_debug_vis()
        # sensors need to be updated
        self.sensor_manager.update()
        self._post_ll_step()
        self.set_observation_buffer()
        self.episode_length_buf += 1
        # print("[INFO][episode_length_buf]{0}".format(self.episode_length_buf))
        # print(self.ll_env.reward_manager.episode_sums["contact_forces"])
        return (self.obs_buf,self.rew_buf, self.reset_buf, self.extras)
    
    def _preprocess_actions(self, actions: torch.Tensor):
        # scale the actions
        self.actions = actions
        self.scaled_action = self.actions * self.action_scale
        self.scaled_action += self.action_offset
        # set the velocity command
        if not self.play_mode_ll:
            self.ll_env.set_velocity_commands(self.scaled_action[:, 0],self.scaled_action[:, 1],self.scaled_action[:, 2])
        # if self.scaled_action.shape[1] > 3:
        #     self.ll_env.set_additional_commands(self.scaled_action[:, 3:])
        self.ll_env.obs_dict = self.ll_env.obs_manager.compute_obs(self.ll_env)
        # self.ll_env.set_observation_buffer()
        # self._update_history_buffers()
        return actions  
    def _get_ll_actions(self):
        """Apply actions to simulation buffers in the environment."""
        self.ll_env._update_commands()
        obs_ll = self.ll_env.obs_manager.compute_obs(self.ll_env)["policy"]
        ll_action = self.ll_policy(obs_ll)
        return ll_action  
    def _draw_hl_debug_vis(self):
        if self.cfg.env.enable_debug_vis:
            # self.sensor_manager.debug_vis(self.ll_env.envs)
            # self._draw_global_memory()
            self._draw_target_position()
    def _post_ll_step(self):
        #update the metrics
        self.command_time_left -= self.dt
        self.total_distance += torch.linalg.norm(self.robot.root_pos_w - self.previous_pos, dim=-1)
        self.previous_pos[:] = self.robot.root_pos_w
        #---Update the Local Navigation Module ---
        
        #Update the goal_error_z_check
        self.goal_error_z_check = torch.norm(self.target_pos[:, :2] - self.robot.root_pos_w[:, :2], dim=1)
        z_diff = torch.abs(self.target_pos[:, 2] - self.robot.root_pos_w[:, 2])
        self.goal_error_z_check[z_diff > 3.0] = torch.inf
        #d
        self.update_history()
        self.update_global_memory()
        #-----------------------------------------
        self.reset_buf[:] = self.termination_manager.check_termination(self)
        self.rew_buf[:] = self.reward_manager.compute_reward(self)
        #-------print reward info---------
        # self.reward_manager.log_info(self, torch.arange(self.num_envs), self.extras)
        # print("[INFO][rew_face_front]{0}".format(self.extras["rew_face_front"]))
        # print("[INFO][rew_goal_tracking_dense_dot]{0}".format(self.extras["rew_goal_tracking_dense_dot"]))
        # print("[INFO][rew_goal_dot]{0}".format(self.extras["rew_goal_dot"]))
        # print("[INFO][rew_goal_position]{0}".format(self.extras["rew_goal_position"]))

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) != 0:
            time_outs = self.termination_manager.time_out_buf.nonzero(as_tuple=False).flatten()
            self.extras["episode"] = {"dist_to_goal": torch.mean(torch.norm(self.pos_target[env_ids] - self.robot.root_pos_w[env_ids], dim=1)),
                                      "death_rate": (len(env_ids) - len(time_outs))/len(env_ids),
                                      "dist_to_goal_timeout": torch.mean(torch.norm(self.pos_target[time_outs] - self.robot.root_pos_w[time_outs], dim=1)), 
                                      "total_distance": torch.mean(self.total_distance[env_ids])}
            self.reward_manager.log_info(self, env_ids, self.extras["episode"])
            self.reset_idx(env_ids)

        self.obs_dict = self.obs_manager.compute_obs(self)
    #--------------2.1 Local Navigation Update ---------------
    def update_history(self):
        # Update Stored Poses to be relative to the agent's current position and orientation
        env_ids = torch.arange(self.num_envs).to(self.device)
        goal_pose = self.target_pos.clone()
        frame_quats = self.robot.root_quat_w
        self.wp_history.update_buffers(self, quats=frame_quats)
        self.pose_history_exp.update_buffers(self, quats=frame_quats)
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        # Update history & graph: Add new pose to the history
        self.wp_history.add_pose(goal_pose, self.robot.root_quat_w, env_ids)
        self.pose_history_exp.add_pose(self.robot.root_pos_w, self.robot.root_quat_w, env_ids)
    def update_global_memory(self):
        #Update the stored node's relative position and orientations with respect to the robot
        env_ids = torch.arange(self.num_envs).to(self.device)
        frame_quats = self.robot.root_quat_w
        self.global_memory.update_buffers(self, env_ids=env_ids, quats=frame_quats)
        #Update global memory: Add new Positions and Associated Features to the global memory
        local_map = self.obs_dict["ext"]  # unused for now
        self.global_memory.add_position_and_feature(env=self,env_ids=Ellipsis,positions=self.robot.root_pos_w,feature=local_map,quats=self.robot.root_quat_w,)
    #--------------3. Reset ---------------
    def reset(self):
        """Reset all environment instances."""
        # reset environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.obs_dict = self.obs_manager.compute_obs(self)
        self.set_observation_buffer()
        # return obs
        return self.obs_buf, self.extras
    def reset_idx(self,env_ids):
        self.ll_env.reset_idx(env_ids)
        self.sensor_manager.update()
        self.command_generator.resample(env_ids)
        #---Reset the Local Navigation Module Buffers---

        #3.1 Reset the Global Memory
        # keep old memory (to train with denser memory)
        clear_memory = (
            torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0) > self.cfg.randomization.keep_memory_prob
        )
        if torch.sum(clear_memory) > 0:
            env_ids_to_clear = env_ids[clear_memory]
            self.global_memory.reset_buffers(self, env_ids_to_clear)
            self.global_memory.update_buffers(self, env_ids_to_clear, quats=self.robot.root_quat_w)
        #3.2 Reset history
        self.wp_history.reset_buffers(env=self, env_ids=env_ids)
        self.wp_history.update_buffers(self, quats=self.robot.root_quat_w)
        self.pose_history_exp.reset_buffers(env=self, env_ids=env_ids)
        self.pose_history_exp.update_buffers(self, quats=self.robot.root_quat_w)

        #-----------------------------------------------
        self.total_distance[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
    #-------- 4. Get/Set functions--------
    def get_observations(self):
        return self.obs_buf, self.extras
    def set_observation_buffer(self):
        self.obs_buf = torch.cat([self.obs_dict['prop'].reshape(self.num_envs, -1),self.obs_dict['ext'].reshape(self.num_envs, -1),self.obs_dict['history'].reshape(self.num_envs, -1),self.obs_dict['memory'].reshape(self.num_envs, -1)], dim=1)
        self.extras["observations"] = self.obs_dict
    def set_ll_velocity_commands(self, x_vel, y_vel, yaw_vel):
        self.ll_env.set_velocity_commands(x_vel, y_vel, yaw_vel)
    #-------- 5. Visualization --------
    def _draw_global_memory(self):
        sphere_geom_graph = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 0, 1))
        height_offset = 0.5

        graph_poses = self.global_memory.all_graph_nodes_abs[0][: self.global_memory.num_nodes[0]]
        for i in range(graph_poses.shape[0]):
            sphere_pose = gymapi.Transform(
                gymapi.Vec3(graph_poses[i, 0], graph_poses[i, 1], graph_poses[i, 2] + height_offset), r=None
            )
            gymutil.draw_lines(sphere_geom_graph, self.gym, self.viewer, self.ll_env.envs[0], sphere_pose)
    def _draw_target_position(self):
        self.sphere_geoms_red = BatchWireframeSphereGeometry(num_spheres=1,radius=0.1, num_lats=4, num_lons=4, pose=None, color=(1, 0, 0))
        self.sphere_geoms_red.draw(self.pos_target[0], self.gym, self.viewer, self.ll_env.envs[0])