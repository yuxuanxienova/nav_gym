# isaac-gym
from isaacgym import gymapi

# python
from typing import List, Optional, Sequence, Tuple
from copy import copy
import re
import numpy as np
import torch

# legged-gym
from nav_gym.nav_legged_gym.common.gym_interface import GymInterface
from nav_gym.nav_legged_gym.common.assets.asset import FileAsset
from nav_gym.nav_legged_gym.common.assets.robots.articulation_cfg import ArticulationCfg
from nav_gym.nav_legged_gym.common.actuators.actuator_cfg import ActuatorCfg
from nav_gym.nav_legged_gym.common.actuators.actuator import Actuator
from nav_gym.nav_legged_gym.common.actuators.actuator_net import ActuatorNetMLP,ActuatorNetLSTM


class Articulation(FileAsset):
    """Helper class for managing articulated assets in simulation.

    An articulation is a fixed or floating based asset with degrees of freedom (DOF), i.e. movable joints.
    The class interfaces with the simulation buffers for retrieving and setting DOF state into simulation.
    The DOFs can have varying joint-level controls which are handled via actuator models (such as an ideal
    drive, DC motor, or actuator networks).

    In addition to the methods of the :cls:`Asset`, the class defines the following main attributes:

    - :func:`set_dof_state`: Set the DOF state of the asset.
    - :func:`get_default_dof_state`: Get the default DOF state (loaded from configuration).
    - :func:`get_random_dof_state`: Get randomly sampled DOF state (specified through configuration).
    - :func:`apply_actions`: Apply actions to the articulation.
    """

    dof_pos: torch.Tensor = None
    """DOF positions of all joints. View of sim buffer of shape: (num_envs, num_dof)."""

    dof_vel: torch.Tensor = None
    """DOF velocities of all joints. View of sim buffer of shape: (num_envs, num_dof)."""

    dof_acc: torch.Tensor = None
    """Previous DOF velocities of all joints (before calling :func:`update_buffers()`). Shape: (num_envs, num_dof)"""

    des_dof_pos: torch.Tensor = None
    """Desired DOF positions targets for all joints. View of interface buffer of shape: (num_envs, num_dof)."""

    des_dof_vel: torch.Tensor = None
    """Desired DOF velocities targets for all joints. View of interface buffer of shape: (num_envs, num_dof)."""

    des_dof_torques: torch.Tensor = None
    """ Desired DOF (joint) Torques (torch.Tensor), shape=(num_envs, num_dof), could be out of actuator limits"""

    dof_torques: torch.Tensor = None
    """Physically possible DOF torques targets for all joints. View of interface buffer of shape: (num_envs, num_dof).

    Note: Based on actuator limits, these are torques computed by clipping :obj:`des_dof_torques`.
    """

    soft_dof_pos_limits: torch.Tensor = None
    """DOF positions limits for all joints. Shape: (num_envs, num_dof, 2)."""

    soft_dof_vel_limits: torch.Tensor = None
    """DOF velocity limits for all joints. Shape: (num_envs, num_dof)."""

    soft_dof_torque_limits: torch.Tensor = None
    """DOF torque limits for all joints. Shape: (num_envs, num_dof)."""

    gear_ratio: torch.Tensor = None
    """ Gear ratio of relating motor torques to dof torques. Default: 1. Only ised with VariableGearRatioActuator. Shape: (num_envs, num_dof)"""

    @property
    def gc(self) -> torch.Tensor:
        """Returns the generalized coordinates for an articulated system.

        Returns:
            torch.Tensor: Generalized coordinates `[root_pos, root_quat, dof_pos]` in simulation world frame.
                Shape=(num_envs, 7 + num_dof).
        """
        return torch.cat[self.root_pos_w, self.root_quat_w, self.dof_pos]

    @property
    def gv(self) -> torch.Tensor:
        """Returns the generalized velocities for an articulated system.

        Returns:
            torch.Tensor: Generalized velocities `[root_lin_vel, root_ang_vel, dof_vel]` in simulation world frame.
                Shape=(num_envs, 6 + num_dof).
        """
        return torch.cat[self.root_lin_vel_w, self.root_ang_vel_w, self.dof_vel]

    def __init__(self, cfg: ArticulationCfg, num_envs: int, gym_iface: GymInterface):
        # initialize parent class
        # note: creates instance of the asset (passed from file)
        super().__init__(cfg, num_envs, gym_iface)
        # note: we reassign cfg here for PyLance to recognize the class object
        self.cfg = cfg

        # store asset properties
        self.num_dof = self.gym.get_asset_dof_count(self._asset)
        self.dof_names = self.gym.get_asset_dof_names(self._asset)
        self._asset_dof_props = self.gym.get_asset_dof_properties(self._asset)

        self.has_dof_torque_sensors = False
        # create buffers for storing information
        self.soft_dof_pos_limits = torch.zeros(self.num_dof, 2, device=self.device)
        self.soft_dof_vel_limits = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.soft_dof_torque_limits = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        # create actuators
        # -- flags used to apply actuator commands into sim
        self._dof_command_type = dict.fromkeys(["position", "velocity", "torque"], False)
        # -- dof indices used to store torque measurements from sim into :obj:`dof_torques`
        self._actuators_torque_sensor_ids = list()
        # -- process actuators configuration
        self._actuators = self._prepare_actuators()

    def init_buffers(self):
        # initialize buffers for root and rigid bodies state
        super().init_buffers()

        # find the DOF index range of this actor
        # note: in simulation, we get buffer of shape (num_envs, total_dofs). This is trick to get
        #   dof slice only around the dofs that belong to the actor
        env_handle = self.gym.get_env(self.sim, 0)  # handle to first environment
        dof_indices = [
            self.gym.find_actor_dof_index(env_handle, self._env_actor_id, dof_name, gymapi.IndexDomain.DOMAIN_ENV)
            for dof_name in self.dof_names
        ]
        dof_start_ind = min(dof_indices)
        dof_stop_ind = max(dof_indices) + 1

        # create views looking at the correct range of indices
        self._dof_state = self.gym_iface.dof_state[:, dof_start_ind:dof_stop_ind, :]
        # -- for commands
        self.dof_torques = self.gym_iface.dof_torque_target[:, dof_start_ind:dof_stop_ind]
        # pos and vel targets are only needed for implicit actuators
        if self._dof_command_type["position"] or self._dof_command_type["velocity"]:
            self.des_dof_pos = self.gym_iface.dof_position_target[:, dof_start_ind:dof_stop_ind]
            self.des_dof_vel = self.gym_iface.dof_velocity_target[:, dof_start_ind:dof_stop_ind]
        # create views over acquired buffers
        # note: this helps in quick access for methods
        self.dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # torque readings of all joints (if enabled)
        if len(self._actuators_torque_sensor_ids) > 0:
            # enable the torque sensors (used when filling up buffers in update call)
            self.has_dof_torque_sensors = True
            # create views looking at the correct range of indices
            self._meas_dof_torque = self.gym_iface.dof_torque[:, dof_start_ind:dof_stop_ind]

        # create buffers
        # -- for history
        self._last_dof_vel = torch.zeros_like(self.dof_vel)  # needed to compute dof_acc
        self.dof_acc = torch.zeros_like(self.dof_vel)

        self._last_body_vel = torch.zeros_like(self.rigid_body_states[:, :, 7:10])  # needed to compute dof_acc #TODO move to asset ?
        self.body_acc = torch.zeros_like(self.rigid_body_states[:, :, 7:10])

        self.des_dof_torques = torch.zeros_like(self.dof_pos)
        self.gear_ratio = torch.ones_like(self.dof_pos)
        # create buffers from config
        # -- process default DOF positions and velocities from cfg
        self.default_dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.default_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        for key, val in self.cfg.init_state.dof_pos.items():
            indices, _ = self.find_dofs(key)
            self.default_dof_pos[:, indices] = val
        for key, val in self.cfg.init_state.dof_vel.items():
            indices, _ = self.find_dofs(key)
            self.default_dof_vel[:, indices] = val

    def spawn(self, env_id: int, pos: Sequence[float] = None, quat: Sequence[float] = None):
        # spawn asset from the parent method
        super().spawn(env_id, pos, quat)
        # THINK: do this once in Isaac interface ?
        env_handle = self.gym.get_env(self.sim, env_id)

        # randomize actor specific properties
        # -- DOF properties
        dof_props = copy(self._asset_dof_props)
        dof_props = self._process_dof_props(dof_props, env_id)
        self.gym.set_actor_dof_properties(env_handle, self._sim_actor_ids[-1], dof_props)

        # enable joint dof force sensor on actor
        # note: this only works if the asset has joints
        if self.cfg.enable_dof_force_sensors:
            self.gym.enable_actor_dof_force_sensors(env_id, self._sim_actor_ids[-1])

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        # reset root-state
        super().reset_buffers(env_ids)
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # reset history
        self._last_dof_vel[env_ids] = self.dof_vel[env_ids]
        self.dof_acc[env_ids] = 0.0

        self._last_body_vel[env_ids] = self.rigid_body_states[env_ids, :, 7:10]
        self.body_acc[env_ids] = 0.0
        # reset actuators
        for actuator in self._actuators:
            actuator.reset(env_ids)

    def apply_actions(self, actions: torch.Tensor):
        for actuator in self._actuators:
            act_dof_ids = actuator.dof_ids
            # compute desired DOF state based on model.
            if actuator.command_type == "P":
                desired_pos = actions[:, actuator.dof_ids] + self.default_dof_pos[:, actuator.dof_ids]
                desired_vel = 0.0
            elif actuator.command_type == "V":
                desired_pos = self.dof_pos[:, act_dof_ids]  # zero torque from p_gains
                desired_vel = actions[:, actuator.dof_ids] + self.default_dof_vel[:, actuator.dof_ids]
            elif actuator.command_type == "PV":
                # TODO test if view order is correct
                actions = actions.view(self.num_envs, -1, 2)
                desired_pos = actions[:, actuator.dof_ids, 0] + self.dof_pos  # zero torque from p_gains
                desired_vel = actions[:, actuator.dof_ids, 1] + self.default_dof_vel
            elif actuator.command_type == "P_rel":
                desired_pos = actions[:, actuator.dof_ids] + self.dof_pos[:, actuator.dof_ids]
                desired_vel = 0.0
            elif actuator.command_type == "V_rel":
                desired_pos = self.dof_pos[:, act_dof_ids]  # zero torque from p_gains
                desired_vel = actions[:, actuator.dof_ids] + self.dof_vel[:, actuator.dof_ids]
            elif actuator.command_type == "T":
                desired_pos = self.dof_pos[:, act_dof_ids]  # zero torque from p_gains
                desired_vel = self.dof_vel[:, act_dof_ids]  # zero torque from d_gains
                self.dof_torques[:, act_dof_ids] = actions[:, act_dof_ids]
            # compute torques for explicit actuators
            if actuator.control_type == "explicit":
                #---------Debug no default pos  added-----------------
                print("[Debug][Articulation][apply_actions] no default pos  added!!!")
                desired_pos = actions[:,actuator.dof_ids]
                desired_vel = 0.0
                #-----------------------------------------------------
                # compute torques explicitly
                actuator.set_dof_state(self.dof_pos[:, act_dof_ids], self.dof_vel[:, act_dof_ids])
                actuator.set_desired_dof_state(desired_pos, desired_vel)
                torques = actuator.compute_torque()
                # -- potentially unrealistic torques
                self.des_dof_torques[:, actuator.dof_ids] = torques
                # -- actual simulation torques are clipped to physical actuator limits
                self.dof_torques[:, actuator.dof_ids] = actuator.clip_torques(torques)
            else:
                # set targets in sim interface tensors
                self.des_dof_pos[:, act_dof_ids] = desired_pos
                self.des_dof_vel[:, act_dof_ids] = desired_vel

            self.gear_ratio[:, act_dof_ids] = actuator.gear_ratio
            self.soft_dof_vel_limits[:, act_dof_ids] = (
                actuator.motor_velocity_limit * actuator.gear_ratio
            )  # can be changed by the actuator
            self.soft_dof_torque_limits[:, act_dof_ids] = (
                actuator.motor_torque_limit * actuator.gear_ratio
            )  # can be changed by the actuator

    def update_buffers(self, dt: float, env_ids: Optional[Sequence[int]] = None):
        # update parent buffers
        super().update_buffers(dt, env_ids)
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # compute dof_acc
        self.dof_acc[env_ids] = (self.dof_vel[env_ids] - self._last_dof_vel[env_ids]) / dt
        self.body_acc[env_ids] = (self.rigid_body_states[env_ids, :, 7:10] - self._last_body_vel[env_ids]) / dt
        # fill-up dof force measurements from simulator
        if self.has_dof_torque_sensors:
            self.dof_torques[env_ids][:, self._actuators_torque_sensor_ids] = self._meas_dof_torque[env_ids][
                :, self._actuators_torque_sensor_ids
            ]

    def update_history(self):
        super().update_history()
        self._last_dof_vel[:] = self.dof_vel[:]
        self._last_body_vel[:] = self.rigid_body_states[:, :, 7:10]

    def find_dofs(self, name_keys, dof_subset=None):
        idx_list = []
        names_list = []
        if dof_subset is None:
            dof_names = self.dof_names
        else:
            dof_names = dof_subset
        if not isinstance(name_keys, list):
            name_keys = [name_keys]
        for i, dof_name in enumerate(dof_names):  # TODO check if we need to sort body names
            for re_name in name_keys:
                if re.match(re_name, dof_name):
                    idx_list.append(i)
                    names_list.append(dof_name)
                    continue
        return idx_list, names_list

    """
    Operations - State.
    """

    def set_dof_state(self, env_ids: Optional[Sequence[int]], dof_pos: torch.Tensor, dof_vel: torch.Tensor):
        """Sets the DOF state (position and velocity) of the actor over selected environment indices.

        Args:
            env_ids (torch.Tensor): Environment indices.
                If :obj:`None`, then all indices are used.
            dof_pos (torch.Tensor): Input DOF position for the actor, shape: (len(env_ids), 1).
            dof_vel (torch.Tensor): Input DOF velocity for the actor, shape: (len(env_ids), 1).
        """
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # set into internal buffers
        # note: this sets the values to the gym interface buffers (since it is a view)
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

    def get_default_dof_state(
        self, env_ids: Optional[Sequence[int]] = None, clone=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the default/initial DOF state (position and velocity) of actor.

        Args:
            env_ids (Optional[Sequence[int]], optional): Environment indices.
                Defaults to None (all environment indices).
            clone (bool, optional): Whether to return a copy or not. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The default/initial DOF position and velocity of the actor.
                Each tensor has shape: (len(env_ids), 1).
        """
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # return copy
        if clone:
            return torch.clone(self.default_dof_pos[env_ids]), torch.clone(self.default_dof_vel[env_ids])
        else:
            return self.default_dof_pos[env_ids], self.default_dof_vel[env_ids]

    def get_random_dof_state(self, env_ids: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns randomly sampled DOF state (position and velocity) of actor.

        Currently, the following sampling is supported:
        - DOF positions:
            - uniform sampling between 0.5 to 1.5 times the default DOF position.
        - DOF velocities:
            - zero.

        Args:
            env_ids (Optional[Sequence[int]], optional): Environment indices.
                Defaults to None (all environment indices).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sampled DOF position and velocity of the actor.
                Each tensor has shape: (len(env_ids), 1).
        """
        dof_pos = self.default_dof_pos[env_ids] #* torch.empty_like(self.dof_pos[env_ids]).uniform_(0.5, 1.5)----TODO SET REANDOM DOF POS??
        dof_vel = self.default_dof_vel[env_ids]
        # return sampled dof state
        return dof_pos, dof_vel

    """
    Helper functions - private.
    """

    def _prepare_actuators(self) -> List[Actuator]:
        # store list of all actuators
        actuators = list()
        # iterate over all actuator configuration
        for actuator_dict in self.cfg.actuators:
            actuator_cfg: ActuatorCfg = actuator_dict["actuator"]
            dof_name_keys = actuator_dict["dof_names"]
            p_gains = actuator_dict.get("p_gains")
            d_gains = actuator_dict.get("d_gains")
            # type-hinting for PyLance
            # if not isinstance(actuator_cfg, ActuatorCfg):
            #     continue
            # check if any actuators specified (?)
            if len(dof_name_keys) == 0:
                continue
            # find actuator DOF names through regex string matching
            dof_indices, dof_names = self.find_dofs(dof_name_keys)
            # create actuator model by loading its class
            actuator: Actuator = eval(actuator_cfg.cls_name)(
                cfg=actuator_cfg, dof_ids=dof_indices, num_envs=self.num_envs, device=self.device
            )
            # add to list of actuators
            actuators.append(actuator)
            # read PD gains from configuration
            # set using actuator.set_command() once it exists
            if p_gains:
                for key, val in p_gains.items():
                    indices, _ = self.find_dofs(key, dof_subset=dof_names)
                    actuator._p_gains[..., indices] = val
            if d_gains:
                for key, val in d_gains.items():
                    indices, _ = self.find_dofs(key, dof_subset=dof_names)
                    actuator._d_gains[..., indices] = val
            # track indices for actuators that are implicit
            # note: The book-keeping is used to recognize for which DOFs we need to read the DOF torque
            #   measurements from simulation. For explicit actuators, the dof-torques should be the same
            #   as the one applied into simulator (however, sometimes it behaves weirdly).
            if actuator.control_type == "implicit" and actuator.enable_torque_sensor:
                self._actuators_torque_sensor_ids.extend(actuator.dof_ids)
            # check which buffers to apply commands
            # note: This stores how individual actuator is commanded at the simulation level.
            #   The book-keeping allows quick-checking in the :func:`_apply_actuator_command`.
            if actuator.control_type == "explicit":
                self._dof_command_type["torque"] = True
            elif actuator.control_type == "implicit":
                if "P" in actuator.command_type:
                    self._dof_command_type["position"] = True
                elif "V" in actuator.command_type:
                    self._dof_command_type["velocity"] = True
                else:
                    raise ValueError(f"Unknown command type for implicit actuator model: {actuator.command_type}.")
            else:
                raise ValueError(f"Unknown control type for actuator model: {actuator.control_type}.")
        # return list of actuators
        return actuators

    def _process_dof_props(self, dof_props: np.ndarray, env_id: int) -> np.ndarray:
        """Store/change/randomize the DOF properties of the asset.

        Note:
            This function is called during spawning of the asset into the environment.

        Args:
            props (numpy.ndarray): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        # Process dof limits from URDF + soft limits from config
        if env_id == 0:
            dof_pos_limits = torch.zeros_like(self.soft_dof_pos_limits)
            for i in range(len(dof_props)):
                # store dof limits from parsed file
                dof_pos_limits[i, 0] = dof_props["lower"][i].item()
                dof_pos_limits[i, 1] = dof_props["upper"][i].item()
                # self.soft_dof_vel_limits[:, i] = dof_props["velocity"][i].item() #This should not be read from the URDF, there it is a hard limit which breaks Physics
                # compute soft limits for DOFs
                dof_mean = (dof_pos_limits[i, 0] + dof_pos_limits[i, 1]) / 2
                dof_range = dof_pos_limits[i, 1] - dof_pos_limits[i, 0]
                #---------------------------------Debug---------------------------------
                print("[Debug][Articulation][_process_dof_props] disable soft limits")
                # self.soft_dof_pos_limits[i, 0] = dof_mean - 0.5 * dof_range * self.cfg.soft_dof_limit_factor
                # self.soft_dof_pos_limits[i, 1] = dof_mean + 0.5 * dof_range * self.cfg.soft_dof_limit_factor
        for actuator in self._actuators:
            # process implicit actuator config
            if actuator.control_type == "implicit":
                # set correct drive mode (0: none, 1: position target, 2: velocity target, 3: effort)
                if "P" in actuator.command_type:
                    drive_mode = gymapi.DOF_MODE_POS
                elif "V" in actuator.command_type:
                    drive_mode = gymapi.DOF_MODE_VEL
                else:
                    drive_mode = gymapi.DOF_MODE_EFFORT
                dof_props["driveMode"][actuator.dof_ids] = drive_mode
                # override stiffness and damping of the URDF values if specified in config
                if actuator._p_gains is not None:
                    dof_props["stiffness"][actuator.dof_ids] = actuator._p_gains[0, :].cpu().numpy()
                if actuator._d_gains is not None:
                    dof_props["damping"][actuator.dof_ids] = actuator._d_gains[0, :].cpu().numpy()
                if actuator.cfg.motor_torque_limit is not None:
                    dof_props["hasLimits"][actuator.dof_ids] = True
                    dof_props["effort"][actuator.dof_ids] = actuator.cfg.motor_torque_limit * actuator.gear_ratio
                else:
                    # remove effort limits since we are taking care of that ourselves
                    if actuator.cfg.motor_torque_limit is not None:
                        dof_props["effort"][actuator.dof_ids] = 1.0e9

        return dof_props
