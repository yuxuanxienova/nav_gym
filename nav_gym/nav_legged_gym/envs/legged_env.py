# isaac-gym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_apply

# python
from copy import deepcopy
import torch
import numpy as np

# legged-gym
from nav_gym.nav_legged_gym.envs.base_env import BaseEnv
from nav_gym.nav_legged_gym.envs.legged_env_config import LeggedEnvCfg
from nav_gym.nav_legged_gym.common.assets.robots.legged_robots.legged_robot import LeggedRobot
from nav_gym.nav_legged_gym.common.sensors.sensors import SensorBase, Raycaster
from nav_gym.nav_legged_gym.utils.math_utils import wrap_to_pi
from nav_gym.nav_legged_gym.common.terrain.terrain import Terrain


class LeggedEnv(BaseEnv):
    robot: LeggedRobot
    cfg: LeggedEnvCfg
    """Environment for locomotion tasks using a legged robot."""

    def __init__(self, cfg: LeggedEnvCfg):
        """Initializes the environment instance.

        Parses the provided config file, calls create_sim() (which creates, simulation,
        terrain and environments), initializes pytorch buffers used during training.

        Args:
            cfg (LeggedEnvCfg): Configuration for the environment.
        """
        # Save some helpful quantities from config to make life easier.
        self.dt = cfg.control.decimation * cfg.gym.sim_params.dt
        self._command_ranges = deepcopy(cfg.commands.ranges)
        self._push_interval = np.ceil(cfg.randomization.push_interval_s / self.dt)

        # initialize the parent
        # note: calls the `create_env` function to create environments.
        super().__init__(cfg)

    """
    Implementation Specifics - Public.
    """

    def _init_external_forces(self):
        self.external_forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
        self.external_torques = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)

    def reset_idx(self, env_ids):
        """Reset environments based on specified indices.

        Calls the following functions on reset:
        - :func:`_reset_robot`: Reset the root state and DOF state of the robot.
        - :func:`_resample_commands`: Resample the goal/command for the task. E.x.: desired velocity command.

        Addition to above, the function fills up episode information into extras and resets buffers.

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
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # -- resample commands
        self._resample_commands(env_ids)
        # self._update_commands()

        self.extras["episode"] = dict()
        self.reward_manager.log_info(self, env_ids, self.extras["episode"])
        self.curriculum_manager.log_info(self, env_ids, self.extras["episode"])
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.termination_manager.time_out_buf

        # Resample disturbances
        external_forces = torch.zeros_like(self.external_forces[env_ids])
        external_torques = torch.zeros_like(self.external_torques[env_ids])
        r = torch.empty(len(env_ids), len(self.robot.feet_indices), 3, device=self.device)
        external_forces[:, 0, :] = r[:, 0, :].uniform_(*self.cfg.randomization.external_force)
        external_torques[:, 0, :] = r[:, 0, :].uniform_(*self.cfg.randomization.external_torque)
        external_forces[:, self.robot.feet_indices, :] = r.uniform_(*self.cfg.randomization.external_foot_force)
        self.external_forces[env_ids] = external_forces[:]
        self.external_torques[env_ids] = external_torques[:]

    """
    Implementation Specifics - Private.
    """

    def _create_envs(self):
        """Design the environment instances."""
        # add terrain instance
        self.terrain = Terrain(gym=self.gym, sim=self.sim,device=self.device, num_envs=self.num_envs)
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
        for _, s in self.sensors.items():
            s.needs_update()
        # rewards, resets, ...
        # -- terminations
        self.reset_buf = self.termination_manager.check_termination(self)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # -- rewards
        self.rew_buf = self.reward_manager.compute_reward(self)
        if len(env_ids) != 0 and self.termination_manager.reset_on_termination:
            # -- update curriculum
            if self._init_done:
                self.curriculum_manager.update_curriculum(self, env_ids)
            # -- reset terminated environments
            self.reset_idx(env_ids)
            # re-update robots for envs that were reset
            self.robot.update_buffers(dt=self.dt, env_ids=env_ids)
            # re-update sensors for envs that were reset
            for _, s in self.sensors.items():
                s.needs_update()

        # update velocity commands
        self._update_commands()

        # Push all robots
        if self.cfg.randomization.push_robots and (self.common_step_counter % self._push_interval == 0):
            self._push_robots()
        # -- obs
        self.obs_dict = self.obs_manager.compute_obs(self)

    def _draw_debug_vis(self):
        """Draws height measurement points for visualization."""
        # draw height lines
        if "height_scanner" not in self.sensors:
            return
        self.sensors["height_scanner"].debug_vis(self)
        # self.sensors["height_scanner2"].debug_vis(self)

    """
    Helper functions (order of calling).
    """

    def _init_buffers(self):
        super()._init_buffers()
        """Initialize torch tensors which will contain simulation states and processed quantities."""
        # initialize some data used later on
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- action buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        # -- command: x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        # assets buffers
        # -- robot
        self.robot.init_buffers()

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

    def _resample_commands(self, env_ids):
        """Randomly select commands of some environments."""
        if len(env_ids) == 0:
            return

        r = torch.empty(len(env_ids), device=self.device)
        # print(self.commands[env_ids], env_ids)
        self.commands[env_ids, 0] = r.uniform_(self._command_ranges.lin_vel_x[0], self._command_ranges.lin_vel_x[1])
        # linear velocity - y direction
        self.commands[env_ids, 1] = r.uniform_(self._command_ranges.lin_vel_y[0], self._command_ranges.lin_vel_y[1])
        # # ang vel yaw - rotation around z
        self.commands[env_ids, 2] = r.uniform_(self._command_ranges.ang_vel_yaw[0], self._command_ranges.ang_vel_yaw[1])
        # heading target
        if self.cfg.commands.heading_command:
            self.heading_target[env_ids] = r.uniform_(self._command_ranges.heading[0], self._command_ranges.heading[1])
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.commands.rel_heading_envs

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.commands.rel_standing_envs

    def _update_commands(self):
        """Sets velocity commands to zero for standing envs, computes angular velocity from heading direction."""
         # check if need to resample
        env_ids = self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        env_ids = env_ids.nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            # Compute angular velocity from heading direction for heading envs
            heading_env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            forward = quat_apply(self.robot.root_quat_w[heading_env_ids, :], self.robot._forward_vec_b[heading_env_ids])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[heading_env_ids, 2] = torch.clip(
                0.5 * wrap_to_pi(self.heading_target[heading_env_ids] - heading),
                self.cfg.commands.ranges.ang_vel_yaw[0],
                self.cfg.commands.ranges.ang_vel_yaw[1],
            )

        # Enforce standing (i.e., zero velocity commands) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.commands[standing_env_ids, :] = 0.0

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        self.robot.root_states[:, 7:13] += torch.empty(self.num_envs, 6, device=self.device).uniform_(*self.cfg.randomization.push_vel)
        self.gym_iface.write_states_to_sim()

    def update_history(self):
        super().update_history()
        self.robot.update_history()
        self.last_actions[:] = self.actions[:]
