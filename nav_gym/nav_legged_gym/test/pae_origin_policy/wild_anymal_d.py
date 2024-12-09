# isaac-gym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_apply
from isaacgym.torch_utils import (
    torch_rand_float,
    quat_rotate_inverse,
    to_torch,
    get_axis_params,
    quat_apply,
)
# python
from copy import deepcopy
import torch
import numpy as np
from typing import Tuple, Union, Dict, Any
import math
import torch
import abc
import json
# legged-gym
from nav_gym.nav_legged_gym.envs.config_locomotion_pae_env import LocomotionPAEEnvCfg
from nav_gym.nav_legged_gym.common.assets.robots.legged_robots.legged_robot import LeggedRobot
from nav_gym.nav_legged_gym.common.sensors.sensors import SensorBase, Raycaster
from nav_gym.nav_legged_gym.utils.math_utils import wrap_to_pi
from nav_gym.nav_legged_gym.common.terrain.terrain_unity import TerrainUnity
from nav_gym.nav_legged_gym.common.terrain.terrainPlane import TerrainPlane
from nav_gym.nav_legged_gym.common.gym_interface import GymInterface
from nav_gym.nav_legged_gym.common.rewards.reward_manager import RewardManager
from nav_gym.nav_legged_gym.common.observations.observation_manager import ObsManager
from nav_gym.nav_legged_gym.common.terminations.termination_manager import TerminationManager
from nav_gym.nav_legged_gym.common.curriculum.curriculum_manager import CurriculumManager
from nav_gym.nav_legged_gym.common.sensors.sensor_manager import SensorManager
from nav_gym.nav_legged_gym.common.commands.command import CommandBase,UnifromVelocityCommand,UnifromVelocityCommandCfg
from nav_gym.nav_legged_gym.utils.visualization_utils import BatchWireframeSphereGeometry
from nav_gym.nav_legged_gym.envs.modules.pae_module import FLD_PAEModule
class WildAnymal:
    cfg: LocomotionPAEEnvCfg

    def __init__(
        self,
        cfg: LocomotionPAEEnvCfg, 
    ):
        #----------------------
        # self.cfg = cfg
        # self.obs_scales = self.cfg.normalization.obs_scales
        # self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        
        # self._init_done = False
        # # copy input arguments into class members
        # self.cfg = cfg
        # # create isaac-interface
        # self.gym_iface = GymInterface(cfg.gym)
        # self.device = self.gym_iface.device
        # # store the environment information from config
        # self.num_envs = self.cfg.env.num_envs
        # self.num_obs = cfg.env.num_observations
        # self.num_privileged_obs = cfg.env.num_privileged_obs
        # """Number of environment instances."""
        # self.num_actions = self.cfg.env.num_actions
        # self.action_space = gym.spaces.Box(
        #     low=-self.cfg.control.action_clipping,
        #     high=self.cfg.control.action_clipping,
        #     shape=(self.num_actions,),
        #     dtype=np.float32,
        # )
        # """Number of actions in the environment."""
        # self.dt = self.cfg.control.decimation * self.cfg.gym.sim_params.dt
        # """Discretized time-step for episode horizon."""
        # self.max_episode_length_s = self.cfg.env.episode_length_s
        # """Maximum duration of episode (in seconds)."""
        # self.max_episode_length = math.ceil(self.max_episode_length_s / self.dt)
        # """Maximum number of steps per episode."""
        #1. Store the environment information from config
        self._init_done = False
        self.cfg = cfg
        #---------------------
        # self.obs_scales = self.cfg.normalization.obs_scales
        # # self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        #-------------------
        self.num_envs = self.cfg.env.num_envs
        self.num_actions = self.cfg.env.num_actions
        self.dt = self.cfg.control.decimation * self.cfg.gym.sim_params.dt
        #Note:
        #simulation loop interval = sim_params.dt (0.005=1/240[s])
        #control loop interval = control.decimation * sim_params.dt (4 * 0.005 = 0.02[s])
        self.max_episode_length_s = self.cfg.env.episode_length_s
        """Maximum duration of episode (in seconds)."""
        self.max_episode_length = math.ceil(self.max_episode_length_s / self.dt)
        """Maximum number of steps per episode."""
        #2. Store other environment information
        self._push_interval = int(np.ceil(cfg.randomization.push_interval_s / self.dt))
        # self._command_ranges = deepcopy(cfg.commands.ranges)

        #3. Create isaac-interface
        self.gym_iface = GymInterface(cfg.gym)
        #4. Create envs, sim
        self.device = self.gym_iface.device
        self.gym = self.gym_iface.gym
        self.sim = self.gym_iface.sim
        self._create_envs()
        #5. Prepare sim buffers
        self.gym_iface.prepare_sim()
        #6. Store commonly used members from gym-interface for easy access.
        self.viewer = self.gym_iface.viewer

        #7. Initialize buffers for environment
        self._init_buffers()
        self.sensors = dict()
        self._init_external_forces()
        # self._prepare_reward_function()

        #8. Prepare mdp helper managers
       
        self.sensor_manager = SensorManager(self)
        self.command_generator: CommandBase = eval(self.cfg.commands.class_name)(self.cfg.commands, self)
        self.reward_manager = RewardManager(self)
        self.fld_module = FLD_PAEModule(self)
        self.obs_manager = ObsManager(self)
        self.termination_manager = TerminationManager(self)
        # self.curriculum_manager = CurriculumManager(self)


        #11. Perform initial reset of all environments (to fill up buffers)
        self.reset()
        self.num_obs = self.obs_buf.shape[1]
        self.num_privileged_obs = None

        #12. Create debug usage
        # self.sphere_geoms_red = BatchWireframeSphereGeometry(num_spheres=1,radius=0.1, num_lats=4, num_lons=4, pose=None, color=(1, 0, 0))
        # self.sphere_geoms_green = BatchWireframeSphereGeometry(num_spheres=1,radius=0.1, num_lats=4, num_lons=4, pose=None, color=(0, 1, 0))
        # self.sphere_geoms_blue = BatchWireframeSphereGeometry(num_spheres=1,radius=0.1, num_lats=4, num_lons=4, pose=None, color=(0, 0, 1))
        #12. Store the flags
        self.flag_enable_reset = True
        self.flag_enable_resample = True
        # we are ready now! :)
        self._init_done = True

    def _init_buffers(self):
        # # allocate common buffers
        # # self.obs_dict = dict()
        # # self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        # self.rew_buf= torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        # self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # # allocate dictionary to store metrics
        # self.extras = dict()
        # # initialize some data used later on
        # # -- counter for curriculum
        # self.common_step_counter = 0
        # # -- action buffers
        # self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        # self.last_actions = torch.zeros_like(self.actions)
        # self.last_last_actions = torch.zeros_like(self.actions)
        # self.processed_actions = torch.zeros_like(self.actions)

        # # -- command: x vel, y vel, yaw vel, heading
        # self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        # # assets buffers
        # # -- robot
        # self.robot.init_buffers()
        # self.last_feet_vel = torch.zeros_like(self.robot.feet_vel)

        # # initialize some data used later on
        # self.extras = {}
        # self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # self.last_dof_vel = torch.zeros_like(self.robot.dof_vel)
        # self.last_root_vel = torch.zeros_like(self.robot.root_states[:, 7:13])
        #----------------------------
        """Initialize torch tensors which will contain simulation states and processed quantities."""
        # allocate common buffers
        self.obs_dict = dict()
        self.rew_buf= torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # allocate dictionary to store metrics
        self.extras = dict()
        # initialize some data used later on
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- action buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.processed_actions = torch.zeros_like(self.actions)
        
        # -- command: x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        # self.heading_target = torch.zeros(self.num_envs, device=self.device)
        # self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        # self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        # assets buffers
        # -- robot
        self.robot.init_buffers()
        self.last_feet_vel = torch.zeros_like(self.robot.feet_vel)
        #----history
        # self.dof_pos_history = torch.zeros(
        #     self.num_envs, 14, self.robot.num_dof, dtype=torch.float, requires_grad=False
        # ).to(self.device)
        # self.dof_vel_history = torch.zeros(
        #     self.num_envs, 14, self.robot.num_dof, dtype=torch.float, requires_grad=False
        # ).to(self.device)

    def _create_envs(self):
        """Design the environment instances."""
        # add terrain instance
        # terrain_curriculum = self.cfg.curriculum.__dict__.get("terrain_levels", None) is not None
        # terrain_generator = TerrainGenerator(self.cfg.terrain, curriculum=False)
        # self.terrain = Terrain(self.cfg.terrain, self.num_envs, self.gym_iface)
        # self.terrain.set_terrain_origins(terrain_generator.terrain_origins)
        # self.terrain.add_mesh(terrain_generator.terrain_mesh, name="terrain")
        # self.terrain.add_to_sim()
        #----------------
        self.terrain = TerrainPlane(self.num_envs, self.gym_iface)
        self.terrain.set_terrain_origins()
        self.terrain.add_to_sim()
        #----------------------------------------------------------
        
        # if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        #     self.custom_origins = True
        # else:
        #     self.custom_origins = False
        #-------------------------------------------------------------
        # add robot class
        robot_cls = eval(self.cfg.robot.cls_name)
        self.robot: LeggedRobot = robot_cls(self.cfg.robot, self.num_envs, self.gym_iface)

        # create environments
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = list()
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)
            # spawn robot
            pos = self.terrain.env_origins[i].clone()
            self.robot.spawn(i, pos)
        #-----------------------------
        #add robot class
        # robot_cls = eval(self.cfg.robot.cls_name)
        # self.robot: LeggedRobot = robot_cls(self.cfg.robot, self.num_envs, self.gym_iface)

        # robot_asset = self.robot._asset
        # self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        # self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        # dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        # rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        # # save body names from the asset
        # body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # self.num_bodies = len(body_names)
        # self.num_dofs = len(self.dof_names)
        # feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        
        # penalized_contact_names = []
        # for name in self.cfg.asset.penalize_contacts_on:
        #     penalized_contact_names.extend([s for s in body_names if name in s])
        # termination_contact_names = []
        # for name in self.cfg.asset.terminate_after_contacts_on:
        #     termination_contact_names.extend([s for s in body_names if name in s])
    
        # base_init_state_list = (
        #     self.cfg.init_state.pos
        #     + self.cfg.init_state.rot
        #     + self.cfg.init_state.lin_vel
        #     + self.cfg.init_state.ang_vel
        # )
        # base_init_state_list = (
        #     self.cfg.robot.init_state.pos
        #     + self.cfg.robot.init_state.rot
        #     + self.cfg.robot.init_state.lin_vel
        #     + self.cfg.robot.init_state.ang_vel
        # )
        # self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        # start_pose = gymapi.Transform()
        # start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # # create environments
        # env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        # env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        # self.envs = list()
        # self.actor_handles = list()
        # for i in range(self.num_envs):
        #     # create env instance
        #     env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
        #     self.envs.append(env_handle)
        #     # spawn robot
        #     pos = self.terrain.env_origins[i].clone()
        #     # self.robot.spawn(i, pos)
        #     pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
        #     start_pose.p = gymapi.Vec3(*pos)

        #     rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
        #     self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
        #     anymal_handle = self.gym.create_actor(
        #         env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0
        #     )
        #     # debugging
        #     self.robot._env_actor_id = anymal_handle
        #     ########
        #     dof_props = self._process_dof_props(dof_props_asset, i)
        #     self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
        #     body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
        #     body_props = self._process_rigid_body_props(body_props, i)
        #     self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
        #     # self.envs.append(env_handle)
        #     self.actor_handles.append(anymal_handle)
        #     if self.cfg.asset.enable_joint_force_sensors:
        #         self.gym.enable_actor_dof_force_sensors(env_handle, anymal_handle)

        # self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(feet_names)):
        #     self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
        #         self.envs[0], self.actor_handles[0], feet_names[i]
        #     )

        # self.penalised_contact_indices = torch.zeros(
        #     len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        # )
        # for i in range(len(penalized_contact_names)):
        #     self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
        #         self.envs[0], self.actor_handles[0], penalized_contact_names[i]
        #     )

        # self.termination_contact_indices = torch.zeros(
        #     len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        # )
        # for i in range(len(termination_contact_names)):
        #     self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
        #         self.envs[0], self.actor_handles[0], termination_contact_names[i]
        #     )

    def _init_external_forces(self):
        self.external_forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
        self.external_torques = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # self.actions = actions.to(self.device)
        # self.actions = self._preprocess_actions(self.actions)
        #----------------------------
        processed_actions = self._preprocess_actions(actions)
        contact_forces = torch.zeros_like(self.robot.net_contact_forces)
        #----------------------------
        # step physics and render each frame
        for _ in range(self.cfg.control.decimation):
            # may include recomputing torques (based on actuator models)
            print("[INFO][step][self.common_step_counter]{0}".format(self.common_step_counter))
            print("[INFO][step][self.actions]{0}".format(self.actions))
            #---------------------------------------
            self._apply_actions(processed_actions)
            # simulation step
            self.gym_iface.simulate()
            # refresh tensors
            self.gym_iface.refresh_tensors(dof_state=True, net_contact_force=True)
            contact_forces = torch.where(
                self.robot.net_contact_forces.norm(dim=2, keepdim=True) > contact_forces.norm(dim=2, keepdim=True),
                self.robot.net_contact_forces,
                contact_forces,
            )
            #sensors update
            self.sensor_manager.update()
            #----------------------------------
            # self._apply_actions(self.actions)
            # self.gym.simulate(self.sim)
            # if self.device == "cpu":
            #     self.gym.fetch_results(self.sim, True)
            # self.gym.refresh_dof_state_tensor(self.sim)

        self.robot.net_contact_forces[:] = contact_forces
         # render viewer
        self.render()
        # update sim counters
        self.episode_length_buf += 1
        # post-physics computation
        self._post_physics_step()
        # return policy obs as the main and rest of observations into extras.
        self.set_observation_buffer()
        # Story memory
        self.update_history()

        # return clipped obs, clipped states (None), rewards, dones and infos
        # clip_obs = self.cfg.normalization.clip_observations
        # self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # if self.privileged_obs_buf is not None:
        #     self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.privileged_obs_buf = None
        return (self.obs_buf,self.rew_buf, self.reset_buf, self.extras)

    
    def _preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Pre-process actions from the environment into actor's commands.
        The step call (by default) performs the following operations:
            - clipping of actions to a range (based on configuration)
            - scaling of actions (based on configuration)
        """
        # processed_actions = self.robot.default_dof_pos + actions * self.cfg.control.action_scale
        # return processed_actions
        #----------------------debug--------------
        # clip actions and move to env device
        actions = torch.clip(actions, -self.cfg.control.action_clipping, self.cfg.control.action_clipping)
        actions = actions.to(self.device)
        # -- default scaling of actions
        scaled_actions = self.cfg.control.action_scale * actions
        self.actions = scaled_actions
        #------------------------------------------
        return scaled_actions

    def _apply_actions(self, actions):
        """Apply actions to simulation buffers in the environment."""
        # set actions to interface buffers
        self.robot.apply_actions(actions)
        # set actions to sim
        self.gym_iface.write_dof_commands_to_sim()

    def render(self, sync_frame_time=True):
        """Render the viewer."""
        # render the GUI
        # perform debug visualization
        self.gym.clear_lines(self.gym_iface.viewer)
        if (
            (self.cfg.env.enable_debug_vis or self.gym_iface.enable_debug_viz)
            and self.gym_iface.viewer
            and self.gym_iface._enable_viewer_sync
        ):
            self._draw_debug_vis()
        else:
            self._stop_debug_vis()
        self.gym_iface.render(sync_frame_time)

    def _draw_debug_vis(self):
        """Draws height measurement points for visualization."""
        #2. draw ray hits
        self.sensor_manager.debug_vis(self.envs)
        # #3. Drawing the Axis
        # sphere_pos_init = self.robot.root_states[:, :3]
        # self.sphere_geoms_red.draw(sphere_pos_init , self.gym, self.viewer, self.envs[0])
        # x_offset = torch.tensor([0.5,0,0],device=self.device)
        # y_offset = torch.tensor([0,0.5,0],device=self.device)
        # z_offset = torch.tensor([0,0,0.5],device=self.device)
        # self.sphere_geoms_red.draw(sphere_pos_init + x_offset, self.gym, self.viewer, self.envs[0])
        # self.sphere_geoms_green.draw(sphere_pos_init + y_offset, self.gym, self.viewer, self.envs[0])
        # self.sphere_geoms_blue.draw(sphere_pos_init + z_offset, self.gym, self.viewer, self.envs[0])

    def _stop_debug_vis(self):
        pass


        
    # def check_termination(self):
    #     """Check if environments need to be reset"""
    #     # self.reset_buf = torch.any(
    #     #     torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
    #     #     dim=1,
    #     # )
    #     self.reset_buf = torch.any(
    #         torch.norm(self.robot.net_contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
    #         dim=1,
    #     )
    #     self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
    #     self.reset_buf |= self.time_out_buf
    
    def _post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        # refresh all tensor buffers
        self.gym_iface.refresh_tensors(
            root_state=True,
            net_contact_force=True,
            rigid_body_state=True,
            dof_state=True,
            dof_torque=self.robot.has_dof_torque_sensors,
        )
        # update env counters (used for curriculum generation)
        self.common_step_counter += 1
        # update robot
        self.robot.update_buffers(dt=self.dt)
        #----update other modules-----
        self.fld_module.on_env_post_physics_step()
        #-----------------------------
        # rewards, resets, ...
        # -- rewards
        self.rew_buf = self.reward_manager.compute_reward(self)
        # -- terminations
        self.reset_buf = self.termination_manager.check_termination(self)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) != 0 and self.termination_manager.reset_on_termination and self.flag_enable_reset:
            # -- update curriculum
            # if self._init_done:
            #     self.curriculum_manager.update_curriculum(self, env_ids)
            # -- reset terminated environments
            self.reset_idx(env_ids)
            # re-update robots for envs that were reset
            self.robot.update_buffers(dt=self.dt, env_ids=env_ids)
            # re-update sensors for envs that were reset
            self.sensor_manager.update()


        # update velocity commands
        self._update_commands()

        # -- obs
        # self.compute_observations()
        self.obs_dict = self.obs_manager.compute_obs(self)
        #-----------------------------------------
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # if self.cfg.asset.enable_joint_force_sensors:
        #     self.gym.refresh_dof_force_tensor(self.sim)

        # self.episode_length_buf += 1
        # self.common_step_counter += 1

        # prepare quantities
        # env_ids = self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        # env_ids = env_ids.nonzero(as_tuple=False).flatten()
        # self._resample_commands(env_ids)

        # compute observations, rewards, resets, ...
        # self.check_termination()

        # self.compute_reward()

        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # self.reset_idx(env_ids)
        # self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)


        # self.last_actions[:] = self.actions[:]
        # self.last_dof_vel[:] = self.robot.dof_vel[:]
        # self.last_root_vel[:] = self.robot.root_states[:, 7:13]
        # self.last_feet_vel[:] = self.feet_vel[:]

        # if self.viewer and self.enable_viewer_sync and self.debug_viz:
        #     self._draw_debug_vis()

    def _update_commands(self):
        """Sets velocity commands to zero for standing envs, computes angular velocity from heading direction."""
         # check if need to resample
        env_ids = self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        env_ids = env_ids.nonzero(as_tuple=False).flatten()
        if self.flag_enable_resample :  
            self.command_generator.resample(env_ids)
            # self._resample_commands(env_ids)
            #--------Other modules--------
            self.fld_module.on_env_resample_commands(env_ids)

        self.command_generator.update()

    def set_velocity_commands(self, x_vel, y_vel, yaw_vel):
        command = (x_vel, y_vel, yaw_vel)
        self.command_generator.set_velocity_command(command)

    # def _resample_commands(self, env_ids):
    #     """Randomly select commands of some environments

    #     Args:
    #         env_ids (List[int]): Environments ids for which new commands are needed
    #     """
    #     self.commands[env_ids, 0] = torch_rand_float(
    #         self.command_ranges["lin_vel_x"][0],
    #         self.command_ranges["lin_vel_x"][1],
    #         (len(env_ids), 1),
    #         device=self.device,
    #     ).squeeze(1)
    #     self.commands[env_ids, 1] = torch_rand_float(
    #         self.command_ranges["lin_vel_y"][0],
    #         self.command_ranges["lin_vel_y"][1],
    #         (len(env_ids), 1),
    #         device=self.device,
    #     ).squeeze(1)
    #     if self.cfg.commands.heading_command:
    #         self.commands[env_ids, 3] = torch_rand_float(
    #             self.command_ranges["heading"][0],
    #             self.command_ranges["heading"][1],
    #             (len(env_ids), 1),
    #             device=self.device,
    #         ).squeeze(1)
    #     else:
    #         self.commands[env_ids, 2] = torch_rand_float(
    #             self.command_ranges["ang_vel_yaw"][0],
    #             self.command_ranges["ang_vel_yaw"][1],
    #             (len(env_ids), 1),
    #             device=self.device,
    #         ).squeeze(1)

    #     # set small commands to zero
    #     self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    # def compute_reward(self):
    #     """Compute rewards
    #     Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
    #     adds each terms to the episode sums and to the total reward
    #     """
    #     self.rew_buf[:] = 0.0
    #     for i in range(len(self.reward_functions)):
    #         name = self.reward_names[i]
    #         rew = self.reward_functions[i]() * self.reward_scales[name]
    #         self.rew[name] = rew / self.dt
    #         self.rew_buf += rew
    #         self.episode_sums[name] += rew
    #     if self.cfg.rewards.only_positive_rewards:
    #         self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
    #     # add termination reward after clipping
    #     if "termination" in self.reward_scales:
    #         rew = self._reward_termination() * self.reward_scales["termination"]
    #         self.rew_buf += rew
    #         self.episode_sums["termination"] += rew
    
    def set_observation_buffer(self):
        # self.obs_buf = torch.cat([self.obs_dict[obs] for obs in self.obs_dict.keys()], dim=1)
        #debug-------
        obs_list = []
        obs_list.append(self.obs_manager.obs_per_func["base_lin_vel"])
        obs_list.append(self.obs_manager.obs_per_func["base_ang_vel"])
        obs_list.append(self.obs_manager.obs_per_func["projected_gravity"])

        obs_list.append(self.obs_manager.obs_per_func["dof_pos"])
        obs_list.append(self.obs_manager.obs_per_func["dof_vel"])
        obs_list.append(self.obs_manager.obs_per_func["actions"])

        obs_list.append(self.obs_manager.obs_per_func["fld_latent_phase_sin"])
        obs_list.append(self.obs_manager.obs_per_func["fld_latent_phase_cos"])
        obs_list.append(self.obs_manager.obs_per_func["fld_latent_onehot"])

        self.obs_buf = torch.cat(obs_list, dim=1)
        #-------------
        self.extras["observations"] = self.obs_dict

    # def compute_observations(self):
    #     """Computes observations"""
    #     self.obs_buf = torch.cat(
    #         (
    #             self.robot.root_lin_vel_b * self.obs_scales.lin_vel,
    #             self.robot.root_ang_vel_b * self.obs_scales.ang_vel,
    #             self.robot.projected_gravity_b,
    #             (self.robot.dof_pos - self.robot.default_dof_pos) * self.obs_scales.dof_pos,
    #             self.robot.dof_vel * self.obs_scales.dof_vel,
    #             # (self.actions - self.default_dof_pos) * self.obs_scales.dof_pos,
    #             self.actions,
    #             # self.feet_pos,
    #         ),

    #         dim=-1,
    #     )
    #     # print("[Debug][self.robot.projected_gravity_b]{0}".format(self.robot.projected_gravity_b))
    #     # print("[Debug][self.projected_gravity]{0}".format(self.projected_gravity))
    #     self.obs_buf = torch.cat((self.obs_buf, torch.sin(2 * torch.pi * self.fld_module.latent_encoding[:, :, 0])), dim=-1)
    #     self.obs_buf = torch.cat((self.obs_buf, torch.cos(2 * torch.pi * self.fld_module.latent_encoding[:, :, 0])), dim=-1)
    #     if self.cfg.fld.one_hot and self.cfg.fld.use_pae:
    #         #motions: Dim: [n_motions*n_trajs, n_slide_win, n_obs_dim, obs_horizon]=[10, 219, 21, 31]
    #         one_hot_encoding = torch.zeros(self.num_envs, self.fld_module.task_sampler.motions.shape[0], device=self.device, requires_grad=False)#Dim: (num_envs, num_motions)
    #         idx = torch.vstack((torch.arange(self.num_envs, device=self.device), self.fld_module.motion_idx)).T
    #         one_hot_encoding[idx[:, 0], idx[:, 1]] = 1
    #         self.obs_buf = torch.cat((self.obs_buf, one_hot_encoding), dim=-1)
        
        # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    # def _get_noise_scale_vec(self, cfg):
    #     noise_vec = torch.zeros_like(self.obs_buf[0])
    #     self.add_noise = self.cfg.noise.add_noise
    #     noise_scales = self.cfg.noise.noise_scales
    #     noise_level = self.cfg.noise.noise_level
    #     noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
    #     noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    #     noise_vec[6:9] = noise_scales.gravity * noise_level
    #     # noise_vec[9:12] = 0.0  # commands
    #     noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
    #     noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
    #     noise_vec[33:45] = 0.0  # previous actions
    #     return noise_vec


    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # self.compute_observations()
        self.obs_dict = self.obs_manager.compute_obs(self)
        self.set_observation_buffer()
        return self.obs_buf, self.extras
    
    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        # -- reset robot state
        self._reset_robot(env_ids)
        # -- write to simulator
        self.gym_iface.write_states_to_sim()

        # -- reset robot buffers
        self.robot.reset_buffers(env_ids)
        # -- reset env buffers
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        #----reset hsitory
        # self.dof_pos_history[env_ids] = 0
        # self.dof_vel_history[env_ids] = 0
        # self.push_robots_buf[env_ids] = torch.randint(
        #     0, self._push_interval, (len(env_ids), 1), device=self.device
        # ).squeeze()

        #--------reset other modules-------
        self.fld_module.on_env_reset_idx(env_ids)
        #----------------------------------
        self.extras["episode"] = dict()
        self.reward_manager.log_info(self, env_ids, self.extras["episode"])
        self.termination_manager.log_info(self, env_ids, self.extras["episode"])
        # self._resample_commands(env_ids)
        # # send timeout info to the algorithm
        # if self.cfg.env.send_timeouts:
        #     self.extras["time_outs"] = self.termination_manager.time_out_buf
        # -- resample commands
        self.command_generator.log_info(self, env_ids, self.extras["episode"])
        self.command_generator.resample(env_ids)
        self.command_generator.update()
        # Resample disturbances
        external_forces = torch.zeros_like(self.external_forces[env_ids])
        external_torques = torch.zeros_like(self.external_torques[env_ids])
        r = torch.empty(len(env_ids), len(self.robot.feet_indices), 3, device=self.device)
        external_forces[:, 0, :] = r[:, 0, :].uniform_(*self.cfg.randomization.external_force)
        external_torques[:, 0, :] = r[:, 0, :].uniform_(*self.cfg.randomization.external_torque)
        external_forces[:, self.robot.feet_indices, :] = r.uniform_(*self.cfg.randomization.external_foot_force)
        self.external_forces[env_ids] = external_forces[:]
        self.external_torques[env_ids] = external_torques[:]

    
    def update_history(self):
        self.robot.update_history()
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_feet_vel[:] = self.robot.feet_vel[:]

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # def _prepare_reward_function(self):
    #     """Prepares a list of reward functions, which will be called to compute the total reward.
    #     Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
    #     """
    #     # remove zero scales + multiply non-zero ones by dt
    #     for key in list(self.reward_scales.keys()):
    #         scale = self.reward_scales[key]
    #         if scale == 0:
    #             self.reward_scales.pop(key)
    #         else:
    #             self.reward_scales[key] *= self.dt
    #     # prepare list of functions
    #     self.reward_functions = []
    #     self.reward_names = []
    #     for name, scale in self.reward_scales.items():
    #         if name == "termination":
    #             continue
    #         self.reward_names.append(name)
    #         name = "_reward_" + name
    #         self.reward_functions.append(getattr(self, name))

    #     # reward episode sums
    #     self.episode_sums = {
    #         name: torch.zeros(
    #             self.num_envs,
    #             dtype=torch.float,
    #             device=self.device,
    #             requires_grad=False,
    #         )
    #         for name in self.reward_scales.keys()
    #     }

    #     self.rew = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) for name in self.reward_scales.keys()}
    #     self.tracking_reconstructed_terms = [name for name in self.reward_scales.keys() if "tracking_reconstructed" in name]

    # def _reward_tracking_reconstructed_lin_vel(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["base_lin_vel"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_lin_vel_scale)

    # def _reward_tracking_reconstructed_ang_vel(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["base_ang_vel"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     print("[INFO][reward_tracking_reconstructed_ang_vel][self.latent_idx]{0}".format(self.fld_module.cur_steps))
    #     print("[INFO][reward_tracking_reconstructed_ang_vel][self.decoded_obs]{0}".format(self.fld_module.target_fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["base_ang_vel"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_ang_vel][self.fld_state]{0}".format(self.fld_module.fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["base_ang_vel"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_ang_vel]{0}".format(torch.exp(-error * self.cfg.rewards.tracking_reconstructed_ang_vel_scale)))
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_ang_vel_scale)

    # def _reward_tracking_reconstructed_projected_gravity(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["projected_gravity"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     print("[INFO][reward_tracking_reconstructed_projected_gravity][self.latent_idx]{0}".format(self.fld_module.cur_steps))
    #     print("[INFO][reward_tracking_reconstructed_projected_gravity][self.decoded_obs]{0}".format(self.fld_module.target_fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["projected_gravity"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_projected_gravity][self.fld_state]{0}".format(self.fld_module.fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["projected_gravity"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_projected_gravity]{0}".format(torch.exp(-error * self.cfg.rewards.tracking_reconstructed_projected_gravity_scale)))
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_projected_gravity_scale)

    # def _reward_tracking_reconstructed_dof_pos_leg_fl(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fl"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fl][self.latent_idx]{0}".format(self.fld_module.cur_steps))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fl][self.decoded_obs]{0}".format(self.fld_module.target_fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fl"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fl][self.fld_state]{0}".format(self.fld_module.fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fl"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fl]{0}".format(torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_leg_fl_scale)))
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_leg_fl_scale)

    # def _reward_tracking_reconstructed_dof_pos_leg_hl(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hl"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hl][self.latent_idx]{0}".format(self.fld_module.cur_steps))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hl][self.decoded_obs]{0}".format(self.fld_module.target_fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hl"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hl][self.fld_state]{0}".format(self.fld_module.fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hl"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hl]{0}".format(torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_leg_hl_scale)))
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_leg_hl_scale)

    # def _reward_tracking_reconstructed_dof_pos_leg_fr(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fr"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fr][self.latent_idx]{0}".format(self.fld_module.cur_steps))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fr][self.decoded_obs]{0}".format(self.fld_module.target_fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fr"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fr][self.fld_state]{0}".format(self.fld_module.fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fr"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fr]{0}".format(torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_leg_fr_scale)))
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_leg_fr_scale)

    # def _reward_tracking_reconstructed_dof_pos_leg_hr(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hr"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hr][self.latent_idx]{0}".format(self.fld_module.cur_steps))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hr][self.decoded_obs]{0}".format(self.fld_module.target_fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hr"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hr][self.fld_state]{0}".format(self.fld_module.fld_state[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hr"], device=self.device, dtype=torch.long, requires_grad=False)]))
    #     print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hr]{0}".format(torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_leg_hr_scale)))
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_dof_pos_leg_hr_scale)
    
    # def _reward_tracking_reconstructed_feet_pos_fl(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["feet_pos_fl"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_feet_pos_fl_scale)
    
    # def _reward_tracking_reconstructed_feet_pos_fr(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["feet_pos_fr"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_feet_pos_fr_scale)
    
    # def _reward_tracking_reconstructed_feet_pos_hl(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["feet_pos_hl"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_feet_pos_hl_scale)
    
    # def _reward_tracking_reconstructed_feet_pos_hr(self):
    #     error = torch.sum(torch.square((self.fld_module.target_fld_state - self.fld_module.fld_state)[:, torch.tensor(self.fld_module.target_fld_state_state_idx_dict["feet_pos_hr"], device=self.device, dtype=torch.long, requires_grad=False)]), dim=1)
    #     return torch.exp(-error * self.cfg.rewards.tracking_reconstructed_feet_pos_hr_scale)


    # def _reward_dof_acc(self):
    #     # Penalize dof accelerations
    #     # return torch.sum(torch.square((self.last_dof_vel - self.robot.dof_vel) / self.dt), dim=1)
    #     return torch.sum(torch.square(self.robot.dof_acc), dim=1)
    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # def _reward_collision(self):
    #     # Penalize collisions on selected bodies
    #     # return torch.sum(
    #     #     1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
    #     #     dim=1,
    #     # )
    #     collision_force = torch.norm(self.robot.net_contact_forces[:, self.penalised_contact_indices, :], dim=-1)
    #     collision_force = torch.where(collision_force > 1., collision_force.clip(min=200.), collision_force) # minimum penalty of 100
    #     return torch.sum(collision_force, dim=1) / 200.
        
    # def _reward_torques(self):
    #     # Penalize torques
    #     # return torch.sum(torch.square(self.torques), dim=1)
    #     return torch.sum(torch.square(self.robot.dof_torques), dim=1)
    
    # def _reward_torque_limits(self):
    #     # penalize torques too close to the limit
    #     # return torch.sum(
    #     #     (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0),
    #     #     dim=1,
    #     return torch.sum(
    #         (torch.abs(self.robot.dof_torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0),
    #         dim=1,
    #     )
    
    # def _reward_feet_acc(self):
    #     """Penalize the feet acceleration when reaching the goal"""
    #     # return torch.sum(torch.square((self.last_feet_vel - self.feet_vel) / self.dt), dim=1)
    #     return torch.sum(torch.square((self.last_feet_vel - self.robot.feet_vel) / self.dt), dim=1)

    # def _reward_termination(self):
    #     # Terminal reward / penalty
    #     return self.reset_buf * ~self.time_out_buf

    def _reset_robot(self, env_ids):
        """Resets root and dof states of robots in selected environments."""
        # -- dof state (handled by the robot)
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids)
        print("[Debug][_reset_robot] disable randomization in dof_pos reset") 
        # dof_pos = dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.robot.set_dof_state(env_ids, dof_pos, dof_vel)
        # -- root state (custom)
        root_state = self.robot.get_default_root_state(env_ids)
        #--------------------
        root_state[:, :3] += self.terrain.env_origins[env_ids]
        # root_state[:, :3] += self.terrain.sample_new_init_poses(env_ids)
        #----------------------
        # shift initial pose
        # root_state[:, :2] += torch.empty_like(root_state[:, :2]).uniform_(
        #     -self.cfg.randomization.max_init_pos, self.cfg.randomization.max_init_pos
        # )
        #-----init root from roll, pitch, yaw--------
        # roll = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.randomization.init_roll_pitch)
        # pitch = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.randomization.init_roll_pitch)
        # yaw = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.randomization.init_yaw)
        # yaw += -np.pi * 2.
        # root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        # root_state[:, 3:7] *= torch.sign(root_state[:, 6]).unsqueeze(1)
        # base velocities: [7:10]: lin vel, [10:13]: ang vel
        #root_state[:, 7:13].uniform_(-0.5, 0.5)
        # set into robot
        self.robot.set_root_state(env_ids, root_state)

    # def _process_rigid_body_props(self, props, env_id):
    #     # randomize base mass
    #     if self.cfg.randomization.randomize_base_mass:
    #         if env_id == 0:
    #             rng = self.cfg.randomization.added_mass_range
    #             self.additional_mass = torch_rand_float(rng[0], rng[1], (self.num_envs, 1), device=self.device)
    #         props[0].mass += self.additional_mass[env_id]
    #     return props
    
    # def _process_dof_props(self, props, env_id):
    #     """Callback allowing to store/change/randomize the DOF properties of each environment.
    #         Called During environment creation.
    #         Base behavior: stores position, velocity and torques limits defined in the URDF

    #     Args:
    #         props (numpy.array): Properties of each DOF of the asset
    #         env_id (int): Environment id

    #     Returns:
    #         [numpy.array]: Modified DOF properties
    #     """
    #     if env_id == 0:
    #         # self.dof_pos_limits = torch.zeros(
    #         #     self.num_dof,
    #         #     2,
    #         #     dtype=torch.float,
    #         #     device=self.device,
    #         #     requires_grad=False,
    #         # )
    #         # self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    #         self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    #         for i in range(len(props)):
    #             # self.dof_pos_limits[i, 0] = props["lower"][i].item()
    #             # self.dof_pos_limits[i, 1] = props["upper"][i].item()
    #             # self.dof_vel_limits[i] = props["velocity"][i].item()
    #             self.torque_limits[i] = props["effort"][i].item()
    #             # soft limits
    #             # m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
    #             # r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
    #             # self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
    #             # self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
    #     return props
    
    # def _process_rigid_shape_props(self, props, env_id):
    #     """Callback allowing to store/change/randomize the rigid shape properties of each environment.
    #         Called During environment creation.
    #         Base behavior: randomizes the friction of each environment

    #     Args:
    #         props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
    #         env_id (int): Environment id

    #     Returns:
    #         [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
    #     """
    #     if self.cfg.randomization.randomize_friction:
    #         if env_id == 0:
    #             # prepare friction randomization
    #             friction_range = self.cfg.randomization.friction_range
    #             num_buckets = 64
    #             bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
    #             friction_buckets = torch_rand_float(
    #                 friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
    #             )
    #             self.friction_coeffs = friction_buckets[bucket_ids]

    #         for s in range(len(props)):
    #             props[s].friction = self.friction_coeffs[env_id]
    #     else:
    #         self.friction_coeffs = torch.ones(self.num_envs, 1, device=self.device)
    #     return props
    
    def get_observations(self) -> torch.Tensor:
        return self.obs_buf, self.extras
    

    def set_flag_enable_reset(self, enable_reset: bool):
        self.flag_enable_reset = enable_reset
        print(f"[INFO][LocomotionEnv]Reset flag set to {enable_reset}")
    def set_flag_enable_resample(self, enable_resample: bool):
        self.flag_enable_resample = enable_resample
        print(f"[INFO][LocomotionEnv]Resample flag set to {enable_resample}")

