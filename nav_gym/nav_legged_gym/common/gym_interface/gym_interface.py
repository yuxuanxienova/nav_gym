from __future__ import annotations

# isaac-gym
from isaacgym import gymapi
from isaacgym import gymutil, gymtorch

# python
from typing import List, Callable, Optional, Iterable, Tuple
from dataclasses import dataclass
import sys
import torch

# legged-gym
from nav_gym.nav_legged_gym.common.gym_interface.gym_interface_cfg import SimParamsCfg, GymInterfaceCfg
from nav_gym.nav_legged_gym.utils.config_utils import class_to_dict


__all__ = ["GymInterface"]


class GymInterface:
    """Interface for IsaacGym.

    This class simplifies handling of IsaacGym simulation context and viewer.
    """

    sim_dt: float
    """Simulation time-step."""
    gravity: torch.Tensor
    """Simulation gravity vector, shape: (3,)."""
    device: str
    """Simulation device to store tensors."""
    gym: "gymapi.Gym"
    """IsaacGym application instance."""
    sim: "gymapi.Sim" = None
    """Current simulation handle of IsaacGym instance."""
    viewer: "gymapi.Viewer" = None
    """Viewer for the simulation."""

    def __init__(self, cfg: Optional[GymInterfaceCfg] = None):
        # replace default value
        if cfg is None:
            print("[WARN]: [GymInterface]: Using default values for gym interface configuration.")
            cfg = GymInterfaceCfg()
        # store inputs
        self.cfg = cfg

        # launch simulation app
        self.gym = gymapi.acquire_gym()
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        # parse configuration
        physics_engine = self._parse_physics_engine(self.cfg)
        self.device, sim_device_id, graphics_device_id = self._parse_devices(self.cfg)
        sim_params = self._parse_sim_params(self.cfg.sim_params)
        # create sim handle
        self.sim = self.gym.create_sim(sim_device_id, graphics_device_id, physics_engine, sim_params)

        # store inputs (for ease reference)
        self.sim_dt = self.cfg.sim_params.dt
        self.gravity = torch.tensor(self.cfg.sim_params.gravity, dtype=torch.float, device=self.device)
        self.headless = self.cfg.headless

        # Needed for headless rendering
        self.has_cameras = False

        # check that simulation is initialized properly
        self._is_prepared = False
        # set all internal tensors to none
        # -- isaac-gym tensors (acquired during @property calls)
        self._dof_state: torch.Tensor = None
        self._dof_torque: torch.Tensor = None
        self._jacobian: torch.Tensor = None
        self._mass_matrix: torch.Tensor = None
        self._root_state: torch.Tensor = None
        self._rigid_body_state: torch.Tensor = None
        self._net_contact_force: torch.Tensor = None
        # -- gym interface tensors (created during @property calls)
        self._dof_position_target: torch.Tensor = None
        self._dof_velocity_target: torch.Tensor = None
        self._dof_torque_target: torch.Tensor = None

        # for smoother camera
        self._enable_viewer_sync = True
        self.enable_debug_viz = False
        self._last_camera_eye_pos: Optional[torch.Tensor] = None
        self._last_camera_target_pos: Optional[torch.Tensor] = None
        # buffers to store data
        self._keyboard_events: List[_KeyboardEvent] = list()
        self.debug_viz_env_idx = 0

    def __del__(self):
        """Cleanup in the end."""
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None

    def prepare_sim(self):
        """Prepare simulation with physics buffers and renderer events allocations.

        Note: This must be called after spawning all the assets into simulation.
        """
        # prepare simulation with buffer allocations (needed for GPU pipeline.)
        self.gym.prepare_sim(self.sim)
        # if running headless handle viewer
        if not self.headless:
            # note: we create viewer here so that if the code fails while setting up the scene, there's no loss.
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            # add basic keyboard shortcuts
            self.register_keyboard_event("QUIT", gymapi.KEY_ESCAPE, self._kbd_quit_event)
            self.register_keyboard_event("toggle_viewer_sync", gymapi.KEY_V, self._kbd_toggle_viewer_sync_event)
            self.register_keyboard_event("toggle_debug_vis", gymapi.KEY_Z, self._kbd_toggle_debug_viz_event)
            self.register_keyboard_event("previous_vis", gymapi.KEY_O, self._kbd_previous_env_viz_event)
            self.register_keyboard_event("next_vis", gymapi.KEY_P, self._kbd_next_env_viz_event)
            # set camera view
            self.set_camera_view(self.cfg.viewer.eye, self.cfg.viewer.target)
        # check what all is present in the sim
        self._sim_dof_count = self.gym.get_sim_dof_count(self.sim)
        self._sim_actor_count = self.gym.get_sim_actor_count(self.sim)
        self._sim_env_count = self.gym.get_env_count(self.sim)

        # Houston, we are ready to simulate!
        self._is_prepared = True
        # print keyboard summary
        if len(self._keyboard_events) > 0:
            print("[INFO]: [GymInterface]: Registered keyboard actions: ")
            for event in self._keyboard_events:
                print(f"\t>>> {event.key}: {event.name}")

    def register_keyboard_event(self, name: str, key: gymapi.KeyboardInput, cb_fn: Callable[[], None]):
        """Register keyboard events into viewer. Used only when running with renderer.

        Args:
            name (str): Name of the keyboard event.
            key (gymapi.KeyboardInput): Associated keyboard key.
            cb_fn (Callable[[], None]): Callback function to call when key pressed.

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        if self.viewer is None:
            if not self.headless:
                print("[WARN]: [GymInterface]: Keyboard actions not available when running in headless mode.")
                return
            else:
                raise RuntimeError(
                    "Keyboard events can only be registered after preparing simulation. Call `prepare_sim()` first."
                )
        # add to existing keyboard events
        event = _KeyboardEvent(name=name, key=key, callback_fn=cb_fn)
        self._keyboard_events.append(event)
        # subscribe to keyboard shortcut
        self.gym.subscribe_viewer_keyboard_event(self.viewer, event.key, event.name)

    def write_states_to_sim(self):
        """Writes the actor root states and DOF states into simulation.

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # deals with case when no actor exists
        if self._sim_actor_count > 0:
            # check if anyone acquired, otherwise set to sim
            if self._root_state is None:
                print("[WARN]: [GymInterface]: Root states must be acquired through GymInterface first. Skipping...")
            else:
                self.gym.set_actor_root_state_tensor(self.sim, self._sim_root_state)
        # deals with case when no DOF exists
        if self._sim_dof_count > 0:
            # check if anyone acquired, otherwise set to sim
            if self._dof_state is None:
                print("[WARN]: [GymInterface]: DOF states must be acquired through GymInterface first. Skipping...")
            else:
                self.gym.set_dof_state_tensor(self.sim, self._sim_dof_state)

    def write_dof_commands_to_sim(self):
        """Writes the actor DOF commands (position targets, velocity targets and effort) into simulation.

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # note: we don't throw warnings here because None is valid :)
        # -- dofs that are position controlled
        if self._dof_position_target is not None:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_position_target))
        # # -- dofs that are velocity controlled
        if self._dof_velocity_target is not None:
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_velocity_target))
        # -- dofs that are effort controlled
        if self._dof_torque_target is not None:
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_torque_target))

    def simulate(self):
        """Step through the physics of the simulation.

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # step through physics
        self.gym.simulate(self.sim)
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

    def refresh_tensors(
        self,
        dof_state: bool = False,
        dof_torque: bool = False,
        jacobian: bool = False,
        mass_matrix: bool = False,
        root_state: bool = False,
        rigid_body_state: bool = False,
        net_contact_force: bool = False,
    ):
        """Updates buffer state for physics related quantities.

        Note:
            This must be called after performing a simulation step to update all buffers and their views
            that are being passed around.

        Args:
            dof_state (bool, optional): Flag for DOF state. Defaults to False.
            dof_torque (bool, optional): Flag for DOF forces. Defaults to False.
            jacobian (bool, optional): Flag for Jacobians. Defaults to False.
            mass_matrix (bool, optional): Flag for mass-matrices. Defaults to False.
            root_state (bool, optional): Flag for actor root state. Defaults to False.
            rigid_body_state (bool, optional): Flag for rigid body state. Defaults to False.
            net_contact_force (bool, optional): Flag for net contact force. Defaults to False.

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # refresh tensors based on flags
        if dof_state:
            self.gym.refresh_dof_state_tensor(self.sim)
        if dof_torque:
            self.gym.refresh_dof_force_tensor(self.sim)
        if jacobian:
            self.gym.refresh_jacobian_tensors(self.sim)
        if mass_matrix:
            self.gym.refresh_mass_matrix_tensors(self.sim)
        if root_state:
            self.gym.refresh_actor_root_state_tensor(self.sim)
        if rigid_body_state:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        if net_contact_force:
            self.gym.refresh_net_contact_force_tensor(self.sim)

    def render(self, sync_frame_time=True):
        """Render the viewer."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit(0)
            # check for keyboard events if triggered
            for kbd_event in self.gym.query_viewer_action_events(self.viewer):
                for event in self._keyboard_events:
                    if kbd_event.action == event.name and kbd_event.value > 0:
                        event.callback_fn()
            # fetch results: populates host buffers from device values
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            # step graphics
            if self._enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                # self.gym.render_all_camera_sensors(self.sim) TODO fix this
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
                if self.has_cameras:
                    self.gym.step_graphics(self.sim)
                    # self.gym.render_all_camera_sensors(self.sim) TODO fix this

        elif self.has_cameras:
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

    def set_camera_view(self, eye: Iterable[float], target: Iterable[float], vel_smoothing: float = 1.0):
        """Positions the viewer camera to look at a specified target position.

        Note:
            This function throws a warning if the viewer is not created. To create viewer, ensure that
            the simulation is running without headless mode and :func:`prepare_sim` has been called.

        Args:
            eye (Iterable[float]): Position of eye of the camera, shape: (3,)
            target (Iterable[float]): Position of target location to look at, shape: (3,)
            vel_smoothing (float, optional): Velocity of the camera to smoothen motion. Range between (0, 1]. Defaults to 1.0.
        """
        # check if viewer exists
        if self.viewer is None:
            print("[WARN]: [GymInterface]: Trying to set camera view when viewer does not exist.")
            return
        # check input is correct
        if vel_smoothing < 0.0 or vel_smoothing > 1.0:
            raise ValueError(f"Expected range of smoothness ratio is '{vel_smoothing}' is outside (0, 1].")
        # convert input into torch tensors
        if not isinstance(eye, torch.Tensor):
            eye = torch.tensor(eye, dtype=torch.float, device=self.device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float, device=self.device)
        # store into internal buffers
        if self._last_camera_eye_pos is None or self._last_camera_target_pos is None:
            self._last_camera_eye_pos = eye
            self._last_camera_target_pos = target
        # smoothen the camera movement (nice for making video/animations)
        if vel_smoothing > 0:
            # compute movement of camera
            if (self._last_camera_eye_pos - eye).norm() < 2.0:
                eye = vel_smoothing * eye + (1.0 - vel_smoothing) * self._last_camera_eye_pos
            if (self._last_camera_target_pos - target).norm() < 2.0:
                target = vel_smoothing * target + (1.0 - vel_smoothing) * self._last_camera_target_pos
        # set into renderer
        # note: we set reference environment position as None (i.e. we use global sim frame)
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(*eye), gymapi.Vec3(*target))
        # store into reference for later
        self._last_camera_eye_pos = eye
        self._last_camera_target_pos = target

    """
    Tensor properties -- From simulation.
    """

    @property
    def dof_state(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: DOF state of all the actors.
                View of sim buffer of shape: (num_envs, total_num_dofs, 2).

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # acquire and reshape the tensor for the first time
        if self._dof_state is None:
            # get gym GPU state as torch.Tensors
            self._sim_dof_state = self.gym.acquire_dof_state_tensor(self.sim)
            self.refresh_tensors(dof_state=True)
            # create views with correct shape
            dof_state = gymtorch.wrap_tensor(self._sim_dof_state)
            self._dof_state = dof_state.view(self._sim_env_count, -1, 2)
            print("[INFO]: [GymInterface]: Simulation DOF state has been acquired.")
        # return the actor root state tensor
        return self._dof_state

    @property
    def dof_torque(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: DOF torques of all the actors.
                View of sim buffer of shape: (num_envs, total_num_dofs).

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # acquire and reshape the tensor for the first time
        if self._dof_torque is None:
            # get gym GPU state as torch.Tensors
            sim_dof_torque = self.gym.acquire_dof_force_tensor(self.sim)
            self.refresh_tensors(dof_torque=True)
            # create views with correct shape
            dof_torque = gymtorch.wrap_tensor(sim_dof_torque)
            self._dof_torque = dof_torque.view(self._sim_env_count, -1)
            print("[INFO]: [GymInterface]: Simulation DOF torque has been acquired.")
        # return the actor root state tensor
        return self._dof_torque

    @property
    def jacobian(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Jacobian of all the actors in simulation world frame.
                View of sim buffer of shape: (num_envs, total_num_bodies, 6, total_num_dofs).

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # acquire and reshape the tensor for the first time
        if self._jacobian is None:
            # get gym GPU state as torch.Tensors
            sim_jacobian = self.gym.acquire_jacobian_tensor(self.sim)
            self.refresh_tensors(jacobian=True)
            # create views with correct shape
            jacobian = gymtorch.wrap_tensor(sim_jacobian)
            self._jacobian = jacobian.view(self._sim_env_count, -1, 6, self._sim_dof_count)
            print("[INFO]: [GymInterface]: Simulation jacobian has been acquired.")
        # return the actor root state tensor
        return self._jacobian

    @property
    def mass_matrix(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Mass matrices of all the actors.
                View of sim buffer of shape: (num_envs, total_num_bodies, total_num_dofs, total_num_dofs).

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # acquire and reshape the tensor for the first time
        if self._mass_matrix is None:
            # get gym GPU state as torch.Tensors
            sim_mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim)
            self.refresh_tensors(mass_matrix=True)
            # create views with correct shape
            mass_matrix = gymtorch.wrap_tensor(sim_mass_matrix)
            self._mass_matrix = mass_matrix.view(self._sim_env_count, -1, self._sim_dof_count, self._sim_dof_count)
            print("[INFO]: [GymInterface]: Simulation mass matrix has been acquired.")
        # return the actor root state tensor
        return self._mass_matrix

    @property
    def root_state(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Root state of all the actors.
                View of sim buffer of shape: (num_envs, num_actors, 13).

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # acquire and reshape the tensor for the first time
        if self._root_state is None:
            # get gym GPU state as torch.Tensors
            self._sim_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.refresh_tensors(root_state=True)
            # create views with correct shape
            root_state = gymtorch.wrap_tensor(self._sim_root_state)
            self._root_state = root_state.view(self._sim_env_count, -1, 13)
            print("[INFO]: [GymInterface]: Simulation root state has been acquired.")
        # return the actor root state tensor
        return self._root_state

    @property
    def rigid_body_state(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Rigid body state of all the actors in simulation world frame.
                View of sim buffer of shape: (num_envs, total_num_bodies, 13).

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # acquire and reshape the tensor for the first time
        if self._rigid_body_state is None:
            # get gym GPU state as torch.Tensors
            sim_rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.refresh_tensors(rigid_body_state=True)
            # create views with correct shape
            rigid_body_state = gymtorch.wrap_tensor(sim_rigid_body_state)
            self._rigid_body_state = rigid_body_state.view(self._sim_env_count, -1, 13)
            print("[INFO]: [GymInterface]: Simulation rigid body state has been acquired.")
        # return the actor root state tensor
        return self._rigid_body_state

    @property
    def net_contact_force(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: DOF torques of all the actors in simulation world frame.
                View of sim buffer of shape: (num_envs, total_num_bodies, 3).

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # acquire and reshape the tensor for the first time
        if self._net_contact_force is None:
            # get gym GPU state as torch.Tensors
            sim_net_contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)
            self.refresh_tensors(net_contact_force=True)
            # create views with correct shape
            net_contact_force = gymtorch.wrap_tensor(sim_net_contact_force)
            self._net_contact_force = net_contact_force.view(self._sim_env_count, -1, 3)
            print("[INFO]: [GymInterface]: Simulation net contact force has been acquired.")
        # return the actor root state tensor
        return self._net_contact_force

    """
    Tensor properties -- From Interface.
    """

    @property
    def dof_position_target(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: DOF position targets for all the actors.
                View of sim buffer of shape: (num_envs, total_num_dofs).

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # create tensor with correct shape
        if self._dof_position_target is None:
            # tensor shape: (num_envs, total_num_dofs)
            dof_position_target = torch.zeros(self._sim_dof_count, dtype=torch.float, device=self.device)
            self._dof_position_target = dof_position_target.view(self._sim_env_count, -1)
            print("[INFO]: [GymInterface]: Simulation desired DOF position has been acquired.")
        # return the actor root state tensor
        return self._dof_position_target

    @property
    def dof_velocity_target(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: DOF velocity targets for all the actors.
                View of sim buffer of shape: (num_envs, total_num_dofs).

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # create tensor with correct shape
        if self._dof_velocity_target is None:
            # tensor shape: (num_envs, total_num_dofs)
            dof_velocity_target = torch.zeros(self._sim_dof_count, dtype=torch.float, device=self.device)
            self._dof_velocity_target = dof_velocity_target.view(self._sim_env_count, -1)
            print("[INFO]: [GymInterface]: Simulation desired DOF velocity has been acquired.")
        # return the actor root state tensor
        return self._dof_velocity_target

    @property
    def dof_torque_target(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: DOF effort targets for all the actors.
                View of sim buffer of shape: (num_envs, total_num_dofs).

        Raises:
            RuntimeError: This function can only be used after calling the function :func:`prepare_sim`.
        """
        # check that sim is prepared
        if not self._is_prepared:
            raise RuntimeError("Simulation only possible after preparing simulation. Call `prepare_sim()` first.")
        # create tensor with correct shape
        if self._dof_torque_target is None:
            # tensor shape: (num_envs, total_num_dofs)
            dof_torque_target = torch.zeros(self._sim_dof_count, dtype=torch.float, device=self.device)
            self._dof_torque_target = dof_torque_target.view(self._sim_env_count, -1)
            print("[INFO]: [GymInterface]: Simulation desired DOF torque has been acquired.")
        # return the actor root state tensor
        return self._dof_torque_target

    """
    Keyboard callback functions.
    """

    def _kbd_quit_event(self):
        # quit with system status as successful.
        print("[INFO]: [GymInterface]: Exiting simulation peacefully...")
        sys.exit(0)

    def _kbd_toggle_viewer_sync_event(self):
        self._enable_viewer_sync = not self._enable_viewer_sync

    def _kbd_toggle_debug_viz_event(self):
        self.enable_debug_viz = not self.enable_debug_viz

    def _kbd_next_env_viz_event(self):
        if self.debug_viz_env_idx < self._sim_env_count-1:
            self.debug_viz_env_idx += 1 
        print(f"[INFO]: [GymInterface]: Switching to environment {self.debug_viz_env_idx} for debug visualization.")

    def _kbd_previous_env_viz_event(self):
        if self.debug_viz_env_idx > 0:
            self.debug_viz_env_idx -= 1 
        print(f"[INFO]: [GymInterface]: Switching to environment {self.debug_viz_env_idx} for debug visualization.")

    """
    Parsing of configuration.
    """

    def _parse_physics_engine(self, cfg: GymInterfaceCfg) -> int:
        # physics engine parameters
        _engine_mapping = {"physx": gymapi.SIM_PHYSX, "flex": gymapi.SIM_FLEX}
        try:
            if isinstance(cfg.physics_engine, str):
                return _engine_mapping[cfg.physics_engine]
            else:
                return cfg.physics_engine
        except KeyError:
            raise ValueError(f"Invalid physics engine: {cfg.physics_engine} not in ['physx', 'flex'].")

    def _parse_devices(self, cfg: GymInterfaceCfg) -> Tuple[str, int, int]:
        # resolve simulation device
        sim_device_type, sim_device_id = gymutil.parse_device_str(cfg.sim_device)
        # device is GPU only if sim is on GPU and use_gpu_pipeline=True
        # otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and cfg.sim_params.use_gpu_pipeline:
            sim_device = cfg.sim_device
        else:
            sim_device = "cpu"
        # graphics device for rendering, -1 for no rendering
        graphics_device_id = cfg.graphics_device_id
        if cfg.headless:
            graphics_device_id = -1

        return sim_device, sim_device_id, graphics_device_id

    def _parse_sim_params(self, cfg: SimParamsCfg) -> gymapi.SimParams:
        gym_sim_params = gymapi.SimParams()
        # set up-axis parameters
        _axis_mapping = {"UP_AXIS_Y": gymapi.UP_AXIS_Y, "UP_AXIS_Z": gymapi.UP_AXIS_Z}
        try:
            if isinstance(cfg.up_axis, str):
                cfg.up_axis = _axis_mapping[cfg.up_axis]
        except KeyError:
            raise ValueError(f"Invalid up-axis in configuration: {cfg.up_axis} not in ['UP_AXIS_Y', 'UP_AXIS_Z'].")
        # set contact collection mode
        _cc_mapping = {0: gymapi.CC_NEVER, 1: gymapi.CC_LAST_SUBSTEP, 2: gymapi.CC_ALL_SUBSTEPS}
        try:
            cfg.physx.contact_collection = _cc_mapping[cfg.physx.contact_collection]
        except KeyError:
            raise ValueError(
                f"Invalid contact collection in configuration: {cfg.physx.contact_collection} not in [0, 1, 2]."
            )
        # set parameters into simulation object
        gymutil.parse_sim_config(class_to_dict(cfg), gym_sim_params)
        # set sim-params based on device
        if self.device != "cpu":
            gym_sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline
            gym_sim_params.physx.use_gpu = cfg.physx.use_gpu
        else:
            gym_sim_params.use_gpu_pipeline = False
            gym_sim_params.physx.use_gpu = False
        # return sim-params
        return gym_sim_params


@dataclass
class _KeyboardEvent:
    """Structure for storing viewer keyboard events."""

    name: str
    """Name of the keyboard event."""
    key: gymapi.KeyboardInput
    """Associated keyboard key."""
    callback_fn: Callable[[], None]
    """Associated keyboard callback function."""
