# isaac-gym
from isaacgym import gymapi, gymtorch
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
from nav_gym.nav_legged_gym.envs.config_locomotion_fld_env import LocomotionEnvCfg
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
from nav_gym.nav_legged_gym.common.commands.command import CommandBase,UnifromVelocityCommand,UnifromVelocityCommandCfg
from nav_gym.nav_legged_gym.utils.visualization_utils import BatchWireframeSphereGeometry
from nav_gym.nav_legged_gym.envs.modules.fld_module import FLDModule
class LocomotionEnv:
    robot: LeggedRobot
    cfg: LocomotionEnvCfg
    """Environment for locomotion tasks using a legged robot."""
#-------- 1. Initialize the environment--------
    def __init__(self, cfg: LocomotionEnvCfg):
        #1. Store the environment information from config
        self._init_done = False
        self.cfg = cfg
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
        self._command_ranges = deepcopy(cfg.commands.ranges)

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

        #8. Prepare mdp helper managers
        self.sensor_manager = SensorManager(self)
        self.command_generator: CommandBase = eval(self.cfg.commands.class_name)(self.cfg.commands, self)
        self.reward_manager = RewardManager(self)
        self.obs_manager = ObsManager(self)
        self.termination_manager = TerminationManager(self)
        self.curriculum_manager = CurriculumManager(self)
        
        #9. Store the environment information from managers
        self.num_obs = self.obs_manager.get_obs_dims_from_group("policy")
        self.num_privileged_obs = self.obs_manager.get_obs_dims_from_group("privileged")
        #10. Initialize Other Modules
        self.fld_module = FLDModule(self)
        #11. Perform initial reset of all environments (to fill up buffers)
        self.reset()
        #12. Create debug usage
        self.sphere_geoms_red = BatchWireframeSphereGeometry(num_spheres=1,radius=0.1, num_lats=4, num_lons=4, pose=None, color=(1, 0, 0))
        self.sphere_geoms_green = BatchWireframeSphereGeometry(num_spheres=1,radius=0.1, num_lats=4, num_lons=4, pose=None, color=(0, 1, 0))
        self.sphere_geoms_blue = BatchWireframeSphereGeometry(num_spheres=1,radius=0.1, num_lats=4, num_lons=4, pose=None, color=(0, 0, 1))
        #12. Store the flags
        self.flag_enable_reset = True
        self.flag_enable_resample = True
        # we are ready now! :)
        self._init_done = True
    def _create_envs(self):
        """Design the environment instances."""
        # add terrain instance
        self.terrain = TerrainUnity(gym=self.gym, sim=self.sim,device=self.device, num_envs=self.num_envs)
        #----------------------------------------------------------
        # terrain_generator = TerrainGenerator(self.cfg.terrain)
        # self.terrain = Terrain(self.cfg.terrain, self.num_envs, self.gym_iface)
        # self.terrain.set_terrain_origins(terrain_generator.terrain_origins)
        # self.terrain.set_valid_init_poses(terrain_generator.valid_init_poses)
        # self.terrain.set_valid_targets(terrain_generator.valid_targets)
        # self.terrain.add_mesh(terrain_generator.terrain_mesh, name="terrain")
        #----------------------------------------------------------
        self.terrain.add_to_sim()
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
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities."""
        # allocate common buffers
        self.obs_dict = dict()
        self.rew_buf = None
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
        # self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        # self.heading_target = torch.zeros(self.num_envs, device=self.device)
        # self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        # self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        # assets buffers
        # -- robot
        self.robot.init_buffers()
        #----history
        self.dof_pos_history = torch.zeros(
            self.num_envs, 14, self.robot.num_dof, dtype=torch.float, requires_grad=False
        ).to(self.device)
        self.dof_vel_history = torch.zeros(
            self.num_envs, 14, self.robot.num_dof, dtype=torch.float, requires_grad=False
        ).to(self.device)
    def _init_external_forces(self):
        self.external_forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
        self.external_torques = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
#-------- 2. Reset the environment--------
    def reset(self) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Reset all environment instances."""
        # reset environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.obs_dict = self.obs_manager.compute_obs(self)
        self.set_observation_buffer()
        # return obs
        return self.obs_buf, self.extras
    def reset_idx(self, env_ids):
        """Reset environments based on specified indices.
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

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
        self.dof_pos_history[env_ids] = 0
        self.dof_vel_history[env_ids] = 0
        # self.push_robots_buf[env_ids] = torch.randint(
        #     0, self._push_interval, (len(env_ids), 1), device=self.device
        # ).squeeze()
        # -- resample commands
        # self._resample_commands(env_ids)
        # self._update_commands()

        self.extras["episode"] = dict()
        self.reward_manager.log_info(self, env_ids, self.extras["episode"])
        self.curriculum_manager.log_info(self, env_ids, self.extras["episode"])
        self.termination_manager.log_info(self, env_ids, self.extras["episode"])
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
    def _reset_robot(self, env_ids):
        """Resets root and dof states of robots in selected environments."""
        # -- dof state (handled by the robot)
        dof_pos, dof_vel = self.robot.get_random_dof_state(env_ids)
        self.robot.set_dof_state(env_ids, dof_pos, dof_vel)
        # -- root state (custom)
        root_state = self.robot.get_default_root_state(env_ids)
        # root_state[:, :3] += self.terrain.env_origins[env_ids]
        root_state[:, :3] += self.terrain.sample_new_init_poses(env_ids)
        # shift initial pose
        # root_state[:, :2] += torch.empty_like(root_state[:, :2]).uniform_(
        #     -self.cfg.randomization.max_init_pos, self.cfg.randomization.max_init_pos
        # )
        roll = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.randomization.init_roll_pitch)
        pitch = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.randomization.init_roll_pitch)
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.randomization.init_yaw)
        # yaw += -np.pi * 2.
        root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        # root_state[:, 3:7] *= torch.sign(root_state[:, 6]).unsqueeze(1)
        # base velocities: [7:10]: lin vel, [10:13]: ang vel
        #root_state[:, 7:13].uniform_(-0.5, 0.5)
        # set into robot
        self.robot.set_root_state(env_ids, root_state)

#-------- 3. Step the environment--------
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        Returns:
            VecEnvStepReturn: A tuple containing:
                - (VecEnvObs) observations from the environment
                - (torch.Tensor) reward from the environment
                - (torch.Tensor) whether the current episode is completed or not
                - (dict) misc information
        """
        #Control loop interval = control.decimation * sim_params.dt (4 * 0.0025 = 0.01[s])
        processed_actions = self._preprocess_actions(actions)
        contact_forces = torch.zeros_like(self.robot.net_contact_forces)
        for _ in range(self.cfg.control.decimation):
            #Simulation loop interval = sim_params.dt (0.0025[s])
            self._apply_actions(processed_actions)
            # apply external disturbance to base and feet
            self._apply_external_disturbance()
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
            # update substep history
            self.update_substep_history()

        self.robot.net_contact_forces[:] = contact_forces
        # render viewer
        self.render()
        # update sim counters
        self.episode_length_buf += 1
        # post-physics computation
        self._post_physics_step()
        # return clipped obs, rewards, dones and infos
        # return policy obs as the main and rest of observations into extras.
        self.set_observation_buffer()
        # Story memory
        self.update_history()
        # return mdp tuples
        return (self.obs_buf,self.rew_buf, self.reset_buf, self.extras)
    def _preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Pre-process actions from the environment into actor's commands.
        The step call (by default) performs the following operations:
            - clipping of actions to a range (based on configuration)
            - scaling of actions (based on configuration)
        """
        # clip actions and move to env device
        actions = torch.clip(actions, -self.cfg.control.action_clipping, self.cfg.control.action_clipping)
        actions = actions.to(self.device)
        self.actions = actions
        # -- default scaling of actions
        scaled_actions = self.cfg.control.action_scale * self.actions
        return scaled_actions
    def _apply_actions(self, actions):
        """Apply actions to simulation buffers in the environment."""
        # set actions to interface buffers
        self.robot.apply_actions(actions)
        # set actions to sim
        self.gym_iface.write_dof_commands_to_sim()
    def _apply_external_disturbance(self):
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.external_forces),
            gymtorch.unwrap_tensor(self.external_torques),
            gymapi.ENV_SPACE,
        )

    def update_substep_history(self):
        if self.num_envs > 1:
            self.dof_pos_history[:, :-1, :] = self.dof_pos_history[:, 1:, :]
            self.dof_vel_history[:, :-1, :] = self.dof_vel_history[:, 1:, :]
            self.dof_pos_history[:, -1, :] = self.robot.dof_pos
            self.dof_vel_history[:, -1, :] = self.robot.dof_vel
        else:
            self.dof_pos_history[:, :-1, :] = self.dof_pos_history[:, 1:, :].clone()
            self.dof_vel_history[:, :-1, :] = self.dof_vel_history[:, 1:, :].clone()
            self.dof_pos_history[:, -1, :] = self.robot.dof_pos
            self.dof_vel_history[:, -1, :] = self.robot.dof_vel

    def render(self, sync_frame_time=True):
        """Render the viewer."""
        # render the GUI
        # perform debug visualization
        # self.gym.clear_lines(self.gym_iface.viewer)
        if (
            (self.cfg.env.enable_debug_vis or self.gym_iface.enable_debug_viz)
            and self.gym_iface.viewer
            and self.gym_iface._enable_viewer_sync
        ):
            self._draw_debug_vis()
        self.gym_iface.render(sync_frame_time)
        #1. clear previous lines
        self.gym.clear_lines(self.viewer)

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
    def _post_physics_step(self):
        """Check terminations, checks erminations and computes rewards, and cache common quantities."""
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
        self.fld_module.update()
        #-----------------------------
        # rewards, resets, ...
        # -- rewards
        self.rew_buf = self.reward_manager.compute_reward(self)
        # -- terminations
        self.reset_buf = self.termination_manager.check_termination(self)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) != 0 and self.termination_manager.reset_on_termination and self.flag_enable_reset:
            # -- update curriculum
            if self._init_done:
                self.curriculum_manager.update_curriculum(self, env_ids)
            # -- reset terminated environments
            self.reset_idx(env_ids)
            # re-update robots for envs that were reset
            self.robot.update_buffers(dt=self.dt, env_ids=env_ids)
            # re-update sensors for envs that were reset
            self.sensor_manager.update()

        # update velocity commands
        self._update_commands()

        # Push all robots
        if self.cfg.randomization.push_robots and (self.common_step_counter % self._push_interval == 0):
            self._push_robots()
        # -- obs
        self.obs_dict = self.obs_manager.compute_obs(self)
    def _update_commands(self):
        """Sets velocity commands to zero for standing envs, computes angular velocity from heading direction."""
         # check if need to resample
        env_ids = self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        env_ids = env_ids.nonzero(as_tuple=False).flatten()
        if self.flag_enable_resample :  
            self.command_generator.resample(env_ids)
        self.command_generator.update()
        self.fld_module.sample_latent_encoding(env_ids)

    def set_velocity_commands(self, x_vel, y_vel, yaw_vel):
        command = (x_vel, y_vel, yaw_vel)
        self.command_generator.set_velocity_commands(command)
    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        self.robot.root_states[:, 7:13] += torch.empty(self.num_envs, 6, device=self.device).uniform_(*self.cfg.randomization.push_vel)
        self.gym_iface.write_states_to_sim()
    def update_history(self):
        self.robot.update_history()
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
#-------- 4. Get/Set functions--------
    def get_observations(self):
        return self.obs_buf, self.extras
    # def get_privileged_observations(self):
    #     return self.obs_manager.get_obs_from_group("privileged")
    def  set_observation_buffer(self):
        self.obs_buf = torch.cat([self.obs_dict[obs] for obs in self.obs_dict.keys()], dim=1)
        self.extras["observations"] = self.obs_dict
    def set_flag_enable_reset(self, enable_reset: bool):
        self.flag_enable_reset = enable_reset
        print(f"[INFO][LocomotionEnv]Reset flag set to {enable_reset}")
    def set_flag_enable_resample(self, enable_resample: bool):
        self.flag_enable_resample = enable_resample
        print(f"[INFO][LocomotionEnv]Resample flag set to {enable_resample}")
#-------- 5. Other functions--------
    def update_learning_curriculum(self):
        pass








