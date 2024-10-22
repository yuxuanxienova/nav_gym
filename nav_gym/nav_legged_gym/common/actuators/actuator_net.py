
# legged-gym
from nav_gym.nav_legged_gym.common.actuators.actuator_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg
from nav_gym.nav_legged_gym.common.actuators.actuator import DCMotor

# python
from typing import List
import torch
class ActuatorNetMLP(DCMotor):
    """Actuator model based on MLP.

    Reference:
        Hwangbo, Jemin, et al. "Learning agile and dynamic motor skills for legged robots."
        Science Robotics 4.26 (2019): eaau5872.
    """

    def __init__(self, cfg: ActuatorNetMLPCfg, dof_ids: List[int], num_envs: int, device: str):
        super().__init__(cfg, dof_ids, num_envs, device)
        # save config locally
        self.cfg = cfg
        # load the model from JIT file
        path = self.cfg.network_file
        self.network = torch.jit.load(path).to(self.device)
        # create buffers for MLP history
        history_length = max(self.cfg.input_idx) + 1
        self._dof_pos_history = torch.zeros(self.num_envs, history_length, self.num_actuators, device=self.device)
        self._dof_vel_history = torch.zeros(self.num_envs, history_length, self.num_actuators, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self._dof_pos_history[env_ids] = 0.0
        self._dof_vel_history[env_ids] = 0.0

    def compute_torque(self) -> torch.Tensor:
        # move history queue by 1 and update top of history
        # -- positions
        self._dof_pos_history = self._dof_pos_history.roll(1, 1)
        self._dof_pos_history[:, 0] = self._des_dof_pos - self._dof_pos
        # -- velocity
        self._dof_vel_history = self._dof_vel_history.roll(1, 1)
        self._dof_vel_history[:, 0] = self._dof_vel

        # compute network inputs
        # -- positions
        pos_input = torch.cat([self._dof_pos_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2)
        pos_input = pos_input.reshape(self.num_envs * self.num_actuators, -1)
        # -- velocity
        vel_input = torch.cat([self._dof_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx], dim=2)
        vel_input = vel_input.reshape(self.num_envs * self.num_actuators, -1)
        # -- concatenate inputs
        network_input = torch.cat([vel_input * self.cfg.vel_scale, pos_input * self.cfg.pos_scale], dim=1)

        # run network inference
        with torch.inference_mode():
            desired_torques = self.network(network_input).reshape(self.num_envs, self.num_actuators)
            desired_torques = self.cfg.torque_scale * desired_torques
        # return torques
        return desired_torques


class ActuatorNetLSTM(DCMotor):
    """LSTM-based actuator model.

    Reference:
        Rudin, Nikita, et al. "Learning to walk in minutes using massively parallel deep reinforcement
        learning." Conference on Robot Learning. PMLR, 2022.
    """

    def __init__(self, cfg: ActuatorNetLSTMCfg, dof_ids: List[int], num_envs: int, device: str):
        super().__init__(cfg, dof_ids, num_envs, device)
        # save config locally
        self.cfg = cfg
        # load the model from JIT file
        path = self.cfg.network_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.network = torch.jit.load(path).to(self.device)

        # extract number of lstm layers and hidden dim from the shape of weights
        num_layers = len(self.network.lstm.state_dict()) // 4
        hidden_dim = self.network.lstm.state_dict()["weight_hh_l0"].shape[1]
        # create buffers
        self.sea_input = torch.zeros(self.num_envs * self.num_actuators, 1, 2, device=self.device)
        self.sea_hidden_state = torch.zeros(
            num_layers, self.num_envs * self.num_actuators, hidden_dim, device=self.device
        )
        self.sea_cell_state = torch.zeros(
            num_layers, self.num_envs * self.num_actuators, hidden_dim, device=self.device
        )

        self.sea_hidden_state_per_env = self.sea_hidden_state.view(
            num_layers, self.num_envs, self.num_actuators, hidden_dim
        )
        self.sea_cell_state_per_env = self.sea_cell_state.view(
            num_layers, self.num_envs, self.num_actuators, hidden_dim
        )

    def reset(self, env_ids: torch.Tensor):
        self.sea_hidden_state_per_env[:, env_ids] = 0.0
        self.sea_cell_state_per_env[:, env_ids] = 0.0

    def compute_torque(self) -> torch.Tensor:
        # compute network inputs
        self.sea_input[:, 0, 0] = (self._des_dof_pos - self._dof_pos).flatten()
        self.sea_input[:, 0, 1] = self._dof_vel.flatten()
        # run network inference
        with torch.inference_mode():
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
            )
        # return torques
        return torques.view(self._dof_pos.shape)
