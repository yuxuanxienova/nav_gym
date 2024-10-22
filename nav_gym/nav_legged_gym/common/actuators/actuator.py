# legged-gym
from nav_gym.nav_legged_gym.common.actuators.actuator_cfg import ActuatorCfg, DCMotorCfg, VariableGearRatioDCMotorCfg

# python
from typing import List
import torch




class Actuator:
    """Ideal actuator model with PD controller and simple saturation model."""

    def __init__(self, cfg: ActuatorCfg, dof_ids: List[int], num_envs: int, device: str):
        # store inputs to class
        self.cfg = cfg
        self.dof_ids = dof_ids
        self.device = device
        self.num_envs = num_envs
        # access config params for quick access
        self.num_actuators = len(dof_ids)
        self.control_type = self.cfg.control_type
        self.command_type = self.cfg.command_type
        self.enable_torque_sensor = self.cfg.enable_torque_sensor

        # create buffers for allocation
        # -- state
        self._dof_pos = torch.zeros(self.num_envs, self.num_actuators, device=device)
        self._dof_vel = torch.zeros_like(self._dof_pos)
        # -- commands
        self._des_dof_pos = torch.zeros_like(self._dof_pos)
        self._des_dof_vel = torch.zeros_like(self._dof_pos)
        # -- PD gains
        self._p_gains = torch.zeros_like(self._dof_pos)
        self._d_gains = torch.zeros_like(self._dof_pos)

        self.gear_ratio = cfg.gear_ratio
        self.motor_torque_limit = self.cfg.motor_torque_limit
        self.motor_velocity_limit = self.cfg.motor_velocity_limit

    def set_dof_state(self, pos: torch.Tensor, vel: torch.Tensor):
        self._dof_pos[:] = pos
        self._dof_vel[:] = vel

    def set_desired_dof_state(self, pos: torch.Tensor, vel: torch.Tensor):
        self._des_dof_pos[:] = pos
        self._des_dof_vel[:] = vel

    def reset(self, env_ids: torch.Tensor):
        pass

    def compute_torque(self) -> torch.Tensor:
        # compute errors
        dof_pos_error = self._des_dof_pos - self._dof_pos
        dof_vel_error = self._des_dof_vel - self._dof_vel
        # return DOF torque
        return self._p_gains * dof_pos_error + self._d_gains * dof_vel_error

    def clip_torques(self, desired_torques) -> torch.Tensor:
        # evaluate parameters for motor
        torque_limit = self.motor_torque_limit * self.gear_ratio
        # saturate tensors
        return torch.clip(desired_torques, -torque_limit, torque_limit)


class DCMotor(Actuator):
    """Direct current motor actuator model with PD controller and velocity-based saturation model."""

    def __init__(self, cfg: DCMotorCfg, dof_ids: List[int], num_envs: int, device: str):
        super().__init__(cfg, dof_ids, num_envs, device)
        # save config locally
        self.cfg = cfg
        self.peak_motor_torque = cfg.peak_motor_torque

    def clip_torques(self, desired_torques) -> torch.Tensor:
        # compute torque limits
        peak_torque = self.peak_motor_torque * self.gear_ratio
        torque_limit = self.motor_torque_limit * self.gear_ratio
        velocity_limit = self.motor_velocity_limit / self.gear_ratio
        max_torques = (peak_torque * (1.0 - self._dof_vel / velocity_limit)).clip(max=torque_limit) #.clip(min=0.0)
        min_torques = (peak_torque * (-1.0 - self._dof_vel / velocity_limit)).clip(min=-torque_limit) #.clip(max=0.0)
        # saturate torques
        torques = torch.min(desired_torques, max_torques)
        torques = torch.max(torques, min_torques)
        return torques


class VariableGearRatioDCMotor(DCMotor):
    """Actuator model with PD controller and gear-ratio based saturation model."""

    def __init__(self, cfg: VariableGearRatioDCMotorCfg, dof_ids: List[int], num_envs: int, device: str):
        super().__init__(cfg, dof_ids, num_envs, device)
        # save config locally
        self.cfg = cfg
        # access quick config
        self.gear_ratio_function = eval(self.cfg.gear_ratio_function)
        self.gear_ratio = torch.ones(self.num_actuators, device=self.device)

    def clip_torques(self, desired_torques) -> torch.Tensor:
        self.gear_ratio = self.gear_ratio_function(self._dof_pos)
        # evaluate parameters for motor
        return super().clip_torques(desired_torques)
