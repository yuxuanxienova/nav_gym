# isaac-gym
from isaacgym.torch_utils import get_axis_params, quat_rotate_inverse, to_torch, quat_apply

# python
import torch
from torch import Tensor

# legged-gym
from nav_gym.nav_legged_gym.common.gym_interface import GymInterface
from nav_gym.nav_legged_gym.common.assets.robots.articulation import Articulation
from nav_gym.nav_legged_gym.common.assets.robots.legged_robots.legged_robots_cfg import LeggedRobotCfg


class LeggedRobot(Articulation):

    feet_pos_w: Tensor = None
    """ Positions of the feet (Tensor), shape=(num_envs, num_feet, 3), view of rigid_body_states"""

    feet_vel_w: Tensor = None
    """ Velocities of the feet (Tensor), shape=(num_envs, num_feet, 3), view of rigid_body_states"""

    feet_current_air_time: Tensor = None
    """ Time since last contact of each foot (Tensor), shape=(num_envs, num_feet)"""

    feet_last_air_time: Tensor = None
    """ Air time before the last contact of each foot (Tensor), shape=(num_envs, num_feet)"""

    forward_vec_w: Tensor = None
    """ Projection of the base forward vector (1, 0, 0) into the world frame (Tensor), shape=(num_envs, 3)"""

    heading_w: Tensor = None
    """ Angle between the x axis of the world and the x axis of the robot (Tensor), shape=(num_envs,)"""

    root_lin_vel_b: Tensor = None
    """ Projection of Root Linear Velocity in base frame (Tensor), shape=(num_envs, 3)"""

    root_ang_vel_b: Tensor = None
    """ Projection of Root Angular Velocity in base frame (Tensor), shape=(num_envs, 3)"""

    projected_gravity_b: Tensor = None
    """ Projection of the Gravity vector in base frame (Tensor), shape=(num_envs, 3)"""

    feet_indices: torch.Tensor
    """ Indices of the feet rigid bodies"""

    contact: torch.Tensor = None
    """ Contact flags """

    def __init__(self, cfg: LeggedRobotCfg, num_envs: int, gym_iface: GymInterface) -> None:
        super().__init__(cfg, num_envs, gym_iface)
        # note: we reassign cfg here for PyLance to recognize the class object
        self.cfg = cfg

    def init_buffers(self):
        super().init_buffers()
        self.feet_indices, _ = self.find_bodies(self.cfg.feet_names)

        self.feet_current_air_time = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.feet_last_air_time = torch.zeros_like(self.feet_current_air_time)
        up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self._gravity_vec_w = to_torch(get_axis_params(-1.0, up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1)
        )
        self._forward_vec_b = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.root_lin_vel_b = quat_rotate_inverse(self.root_quat_w, self.root_lin_vel_w)
        self.root_ang_vel_b = quat_rotate_inverse(self.root_quat_w, self.root_ang_vel_w)
        self.projected_gravity_b = quat_rotate_inverse(self.root_quat_w, self._gravity_vec_w)
        self.forward_vec_w = quat_apply(self.root_quat_w, self._forward_vec_b)
        self.heading_w = torch.zeros(self.num_envs, device=self.device)

        self.feet_pos_w = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel_w = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.feet_pos_global = self.feet_pos_w - self.root_pos_w.unsqueeze(1)
        self.feet_vel_global = self.feet_vel_w - self.root_lin_vel_w.unsqueeze(1)

        self.feet_pos_b = torch.zeros_like(self.feet_pos_w)
        self.feet_vel_b = torch.zeros_like(self.feet_vel_w)
        for i in range(len(self.feet_indices)):
            self.feet_pos_b[:, i] = quat_rotate_inverse(
                self.root_quat_w,
                self.feet_pos_global[:, i]
            )
            self.feet_vel_b[:, i] = quat_rotate_inverse(
                self.root_quat_w, 
                self.feet_vel_global[:, i]
            )
        self.feet_pos = self.feet_pos_b.flatten(1, 2)
        self.feet_vel = self.feet_vel_b.flatten(1, 2)

        self.contact = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )

    def reset_buffers(self, env_ids):
        super().reset_buffers(env_ids)
        self.feet_current_air_time[env_ids] = 0.0
        self.feet_last_air_time[env_ids] = 0.0
        self.forward_vec_w[env_ids] = quat_apply(self.root_quat_w[env_ids], self._forward_vec_b[env_ids])
        self.heading_w[env_ids] = torch.atan2(self.forward_vec_w[env_ids, 1], self.forward_vec_w[env_ids, 0])

    def update_buffers(self, dt: float, env_ids=None):
        super().update_buffers(dt, env_ids)
        if env_ids is None:
            env_ids = ...  # all elements of the tensor

        self.root_lin_vel_b[env_ids] = quat_rotate_inverse(self.root_quat_w[env_ids], self.root_lin_vel_w[env_ids])
        self.root_ang_vel_b[env_ids] = quat_rotate_inverse(self.root_quat_w[env_ids], self.root_ang_vel_w[env_ids])
        self.projected_gravity_b[env_ids] = quat_rotate_inverse(self.root_quat_w[env_ids], self._gravity_vec_w[env_ids])
        self.forward_vec_w[env_ids] = quat_apply(self.root_quat_w[env_ids], self._forward_vec_b[env_ids])
        self.heading_w[env_ids] = torch.atan2(self.forward_vec_w[env_ids, 1], self.forward_vec_w[env_ids, 0])

        self.contact = self.net_contact_forces[:, self.feet_indices, 2] > 1.0
        first_contact = (self.feet_current_air_time > 0.0) * self.contact
        self.feet_current_air_time += dt
        self.feet_last_air_time = self.feet_current_air_time * first_contact
        self.feet_current_air_time *= ~self.contact

        self.feet_pos_w[env_ids, ...] = self.rigid_body_states[:, self.feet_indices, 0:3][env_ids, ...]
        self.feet_vel_w[env_ids, ...] = self.rigid_body_states[:, self.feet_indices, 7:10][env_ids, ...]

        self.feet_pos_global = self.feet_pos_w - self.root_pos_w.unsqueeze(1)
        self.feet_vel_global = self.feet_vel_w - self.root_lin_vel_w.unsqueeze(1)

        self.feet_pos_b = torch.zeros_like(self.feet_pos_w)
        self.feet_vel_b = torch.zeros_like(self.feet_vel_w)
        for i in range(len(self.feet_indices)):
            self.feet_pos_b[:, i] = quat_rotate_inverse(
                self.root_quat_w,
                self.feet_pos_global[:, i]
            )
            self.feet_vel_b[:, i] = quat_rotate_inverse(
                self.root_quat_w, 
                self.feet_vel_global[:, i]
            )
        self.feet_pos = self.feet_pos_b.flatten(1, 2)
        self.feet_vel = self.feet_vel_b.flatten(1, 2)

        self.contact = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )