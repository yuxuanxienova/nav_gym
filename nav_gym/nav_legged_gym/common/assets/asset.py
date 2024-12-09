import re

# isaac-gym
from isaacgym.torch_utils import to_torch
from isaacgym import gymapi, gymtorch

# python
from typing import Optional, Sequence, List, Tuple, Union
import abc
from copy import copy
import numpy as np
import torch
import os

# legged-gym
from nav_gym.nav_legged_gym.common.gym_interface import GymInterface
from nav_gym.nav_legged_gym.common.assets.asset_cfg import AssetCfg,FileAssetCfg


__all__ = ["CuboidAsset", "SphereAsset", "FileAsset"]


class Asset:
    """Helper class for managing assets in simulation.

    An asset is any actor/entity imported into the simulation stage. These include robots, rigid objects or articulated objects.
    It contains attributes corresponding to proprioceptive information. This includes joint information, actor's bodies state,
    contact information and DOF state.

    The class is defined by the following main attributes:

    - :func:`spawn` : Import an instance of the asset into environment instance.
    - :func:`init_buffers` : Once all assets are imported, create views over simulation buffers for asset specific quantities.
    - :func:`set_root_state` : Set the root state of the asset.
    - :func:`get_default_root_state` : Get the default root state (loaded from configuration).
    - :func:`reset_buffers` : At episodic resets, reset values of internal buffers.
    - :func:`update_buffers` : After simulation step, recompute internal buffers.
    """

    root_states: torch.Tensor = None
    """Root state `[pos, quat, lin_vel, ang_vel]` in simulation world frame. View of sim buffer of shape: (num_envs, 13)."""

    last_root_states: torch.Tensor = None
    """Previous root states `[pos, quat, lin_vel, ang_vel]` in simulation world frame. Shape: (num_envs, 13)."""

    rigid_body_states: torch.Tensor = None
    """State of all rigid bodies of the asset `[pos, quat, lin_vel, ang_vel]` in simulation world frame. View of sim buffer of shape: (num_envs, num_bodies, 13)."""

    net_contact_forces: torch.Tensor = None
    """Net contact forces acting on each rigid body in simulation world frame. View of sim buffer of shape: (num_envs, num_bodies, 3)."""

    root_pos_w: torch.Tensor = None
    """Root position in simulation world frame. View of :obj:`root_states` of shape: (num_envs, 3)"""

    root_quat_w: torch.Tensor = None
    """Root quaternion orientation `(x, y, z, w)` in simulation world frame. View of :obj:`root_states` of shape: (num_envs, 4)."""

    root_lin_vel_w: torch.Tensor = None
    """Root linear velocity in simulation world frame. View of :obj:`root_states` of shape: (num_envs, 3)."""

    root_ang_vel_w: torch.Tensor = None
    """Root angular velocity in simulation world frame. View of :obj:`root_states` of shape: (num_envs, 3)."""

    def __init__(self, cfg: AssetCfg, num_envs: int, gym_iface: GymInterface):
        # store members internally.
        self.cfg = cfg
        self.num_envs = num_envs
        self.gym_iface = gym_iface
        # store commonly used members from gym-interface for easy access.
        self.gym = gym_iface.gym
        self.sim = gym_iface.sim
        self.device = gym_iface.device

        # read asset options from config
        asset_options = gymapi.AssetOptions()
        # TODO: Make this function cleaner by using key-matching and setattr.
        asset_options.default_dof_drive_mode = self.cfg.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.fix_base_link
        asset_options.density = self.cfg.density
        asset_options.angular_damping = self.cfg.angular_damping
        asset_options.linear_damping = self.cfg.linear_damping
        asset_options.max_angular_velocity = self.cfg.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.max_linear_velocity
        asset_options.armature = self.cfg.armature
        asset_options.thickness = self.cfg.thickness
        asset_options.disable_gravity = self.cfg.disable_gravity
        # load asset instance into gym
        self._asset = self._retrieve_asset_instance(asset_options)

        # store asset properties
        self._asset_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self._asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self._asset)
        self.body_names = self.gym.get_asset_rigid_body_names(self._asset)

        # create buffers for storing information
        # -- ids of all instances of this asset in simulation
        self._sim_actor_ids = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # -- id of this asset in each env (must be identical in all envs)
        self._env_actor_id = None

        # prepare properties randomization
        self._prepare_randomization_properties()

    def spawn(self, env_id: int, pos: Sequence[float] = None, quat: Sequence[float] = None):
        """Spawn the asset into simulation environment.

        Based on the configuration, the function performs the following additional operations:
        - randomizes the friction material assigned to the actor
        - randomizes the added mass on the specified rigid bodies of the actor
        - randomizes the color of the rigid bodies of the actor

        Note:
            We set the collision filter and collision group of the asset based on the environment
            index (argument to function) and the self-collision flag (from asset configuration).

        Args:
            env_id (int): The environment index.
            pos (Sequence[float], optional): Position in simulation world frame. Defaults to None.
            quat (Sequence[float], optional): Orientation in simulation world frame. Defaults to None.

        Raises:
            ValueError: Asset has an invalid handle.
        """
        #  THINK: do this once in Isaac interface ?
        env_handle = self.gym.get_env(self.sim, env_id)
        # set initial pose when asset is spawned to environment
        start_pose = gymapi.Transform()
        if pos is None:
            pos = self.cfg.init_state.pos
        if quat is None:
            quat = self.cfg.init_state.rot
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(*quat)

        # TODO: generalize -> randomization manager ?
        # modify shape properties
        # TODO: test if copy is actually copying!
        rigid_shape_props = copy(self._asset_rigid_shape_props)
        rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props, env_id)
        self.gym.set_asset_rigid_shape_properties(self._asset, rigid_shape_props)

        # load params from cfg
        asset_name = self.cfg.asset_name
        self_collision = int(not self.cfg.self_collisions)
        seg_id = int(self.cfg.segmentation_id)
        # spawn actor
        actor_id = self.gym.create_actor(
            env_handle, self._asset, start_pose, asset_name, env_id, self_collision, seg_id
        )
        # check valid actor
        if self._env_actor_id is not None and self._env_actor_id != actor_id:
            raise ValueError(f"An asset must have the same actor index in all envs: {self._env_actor_id} != {actor_id}")
        # store information about actor
        self._env_actor_id = actor_id
        self._sim_actor_ids[env_id] = self.gym.get_actor_index(env_handle, 0, gymapi.IndexDomain.DOMAIN_SIM)

        # randomize actor-specific properties
        # -- body properties
        body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_id)
        body_props = self._process_rigid_body_props(body_props, env_id)
        # TODO: check if recompute inertia uses density of the asset or volume to recompute density?
        self.gym.set_actor_rigid_body_properties(env_handle, actor_id, body_props, recomputeInertia=True)
        # -- body color
        if self._visual_colors is not None:
            for body_id in range(self.num_bodies):
                color = gymapi.Vec3(*self._visual_colors[env_id, body_id])
                self.gym.set_rigid_body_color(env_handle, actor_id, body_id, gymapi.MESH_VISUAL_AND_COLLISION, color)

    def init_buffers(self):
        """Initializes buffers for the asset.

        This function acquires tensors from the simulation and creates views over them based on the actor
        indices. Additionally, it creates new tensors for internal buffers such as for history and converting
        configuration parameters into torch tensors.

        Note:
            This function must be called after creating all the environment instances.
        """
        # find the body index range of this actor
        # note: in simulation, we get buffer of shape (num_envs, total_bodies). This is trick to get
        #   body slice only around the bodies that belong to the actor
        env_handle = self.gym.get_env(self.sim, 0)  # handle to first environment
        body_indices = [
            self.gym.find_actor_rigid_body_index(
                env_handle, self._env_actor_id, body_name, gymapi.IndexDomain.DOMAIN_ENV
            )
            for body_name in self.body_names
        ]
        body_start_ind = min(body_indices)
        body_stop_ind = max(body_indices) + 1

        # create views looking at the correct range of indices
        # -- state of the asset's actor root
        self.root_states = self.gym_iface.root_state[:, self._env_actor_id, :]
        # -- state of the asset's rigid bodies
        self.rigid_body_states = self.gym_iface.rigid_body_state[:, body_start_ind:body_stop_ind, :]
        # -- contact forces on all asset's rigid shapes
        self.net_contact_forces = self.gym_iface.net_contact_force[:, body_start_ind:body_stop_ind, :]
        # create views over acquired buffers
        # note: this helps in quick access for methods
        self.root_pos_w = self.root_states[:, :3]
        self.root_quat_w = self.root_states[:, 3:7]
        self.root_lin_vel_w = self.root_states[:, 7:10]
        self.root_ang_vel_w = self.root_states[:, 10:13]

        # create buffers for storing history
        self.last_root_states = torch.zeros_like(self.root_states)
        # create buffers from config
        default_root_state = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self._default_root_states = to_torch(default_root_state, device=self.device).repeat(self.num_envs, 1)

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        """Reset internal buffers based on selected environments.

        Note:
            This method must be called after setting actor's state via :func:`set_` methods.

        Args:
            env_ids (Optional[Sequence[int]], optional): Environment indices.
                Defaults to None (all environment indices).
        """
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # reset history
        self.last_root_states[env_ids] = self.root_states[env_ids]

    def update_buffers(self, dt: float, env_ids: Optional[Sequence[int]] = None):
        """Update the internal buffers based on selected environments.

        Note:
            This method must be called after performing simulation step.

        Args:
            dt (float): time-step [s]
            env_ids (Sequence[int], optional): Environment indices.
                Defaults to None (all environment indices).
        """
        pass

    def find_bodies(self, name_keys):
        idx_list = []
        names_list = []
        if not isinstance(name_keys, list):
            name_keys = [name_keys]
        for i, body_name in enumerate(self.body_names):  # TODO check if we need to sort body names
            for re_name in name_keys:
                if re.match(re_name, body_name):
                    idx_list.append(i)
                    names_list.append(body_name)
                    continue
        return idx_list, names_list

    """
    Operations - State.
    """

    def update_history(self):
        self.last_root_states[:] = self.root_states[:]

    def set_root_state(self, env_ids: Optional[Sequence[int]], root_states: torch.Tensor):
        """Sets the root state (pose and velocity) of the actor over selected environment indices.

        Args:
            env_ids (Optional[Sequence[int]]): Environment indices.
                If :obj:`None`, then all indices are used.
            root_states (torch.Tensor): Input root state for the actor, shape: (len(env_ids), 13).
        """
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # set into internal buffers
        # note: this sets the values to the gym interface buffers (since it is a view)
        self.root_states[env_ids] = root_states

    def get_default_root_state(self, env_ids: Optional[Sequence[int]] = None, clone=True) -> torch.Tensor:
        """Returns the default/initial root state of actor.

        Args:
            env_ids (Optional[Sequence[int]], optional): Environment indices.
                Defaults to None (all environment indices).
            clone (bool, optional): Whether to return a copy or not. Defaults to True.

        Returns:
            torch.Tensor: The default/initial root state of the actor, shape: (len(env_ids), 13).
        """
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # return copy
        if clone:
            return torch.clone(self._default_root_states[env_ids])
        else:
            return self._default_root_states[env_ids]

    """
    Helper functions.
    """

    def print_asset_info(self):
        """Print information about the asset object such as number of bodies, joints and DOFs."""
        num_bodies = self.gym.get_asset_rigid_body_count(self._asset)
        num_joints = self.gym.get_asset_joint_count(self._asset)
        num_dofs = self.gym.get_asset_dof_count(self._asset)
        # Iterate through bodies
        if num_bodies > 0:
            print(">> Bodies:")
            for i in range(num_bodies):
                body_name = self.gym.get_asset_rigid_body_name(self._asset, i)
                print(f"\t Index {i}: {body_name}")
        # Iterate through joints
        if num_joints > 0:
            print(">> Joints:")
            for i in range(num_joints):
                name = self.gym.get_asset_joint_name(self._asset, i)
                joint_type = self.gym.get_asset_joint_type(self._asset, i)
                joint_type_name = self.gym.get_joint_type_string(joint_type)
                print(f"\t Index {i}: '{name}' ({joint_type_name})")
        # iterate through degrees of freedom (DOFs)
        if num_dofs > 0:
            print(">> DOFs:")
            for i in range(num_dofs):
                name = self.gym.get_asset_dof_name(self._asset, i)
                dof_type = self.gym.get_asset_dof_type(self._asset, i)
                dof_type_name = self.gym.get_dof_type_string(dof_type)
                print(f"\t Index {i}: '{name}' ({dof_type_name})")

    def attach_camera(
        self,
        env_id: int,
        body_name: str,
        local_pose: gymapi.Transform,
        camera_properties: gymapi.CameraProperties,
        image_type: int,
    ) -> Tuple[Tuple[torch.Tensor], Tuple[int]]:
        """Attaches a camera to the actor.

        Args:
            body_name (str): The name of the body to attach the camera to.
            local_pose (gymapi.Transform): The local pose with respect to the body.
            camera_properties (gymapi.CameraProperties): The camera properties.
            image_type (Union[gymapi.IMAGE_DEPTH, gymapi.IMAGE_COLOR]): The image type.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the image tensors.
            Tuple[int]: A tuple containing the camera handles.
        """
        env_handle = self.gym.get_env(self.sim, env_id)
        body_idx, _ = self.find_bodies(body_name)
        self.gym_iface.has_cameras = True
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_properties)
        self.gym.attach_camera_to_body(camera_handle, env_handle, body_idx[0], local_pose, gymapi.FOLLOW_TRANSFORM)
        camera_buffer = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, camera_handle, image_type)
        camera_tensor = gymtorch.wrap_tensor(camera_buffer)
        return camera_tensor, camera_handle

    """
    Helper functions - private.
    """

    @abc.abstractmethod
    def _retrieve_asset_instance(self, asset_options: gymapi.AssetOptions) -> gymapi.Asset:
        """Helper function to import asset based on its type.

        Args:
            asset_options (gymapi.AssetOptions): The settings for imported asset.

        Returns:
            gymapi.Asset: The imported asset instance in the simulation.
        """
        raise NotImplementedError("This class should not be used. Check its children under inheritance.")

    """
    Helper functions.
    """

    def _prepare_randomization_properties(self):
        """Prepare randomization properties from sampling ranges.

        This function pre-computes the friction and visual materials based on configuration.
        - `friction_coeffs`: friction material for each actor instance, shape: (num_envs, 1).
        - `visual_colors`: visual color for each body of actor, shape: (num_envs, num_bodies, 3).
        """
        # -- friction
        if self.cfg.randomization.randomize_friction:
            friction_range = self.cfg.randomization.friction_range
            num_buckets = self.cfg.randomization.friction_buckets
            friction_buckets = torch.empty(num_buckets).uniform_(friction_range[0], friction_range[1])
            bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
            self._friction_coeffs = friction_buckets[bucket_ids]
        else:
            self._friction_coeffs = None
        # -- added mass to rigid bodies
        if self.cfg.randomization.randomize_added_mass:
            rng = self.cfg.randomization.added_mass_range
            indices = self.cfg.randomization.added_mass_rigid_body_indices
            # sample added mass from range
            self._added_masses = np.random.uniform(rng[0], rng[1], size=(self.num_envs, len(indices)))
        else:
            self._added_masses = None
        # if self.cfg.randomization.randomize_restitution:
        #     restitution_range = self.cfg.randomization.restitution_range
        #     num_buckets = self.cfg.randomization.restitution_buckets
        #     restitution_buckets = torch.empty(num_buckets).uniform_(restitution_range[0], restitution_range[1])
        #     bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
        #     self._restitution_coeffs = restitution_buckets[bucket_ids]
        # -- color
        if self.cfg.randomization.randomize_color == 0:
            self._visual_colors = None
        elif self.cfg.randomization.randomize_color == 1:
            color = self.cfg.randomization.color_fixed
            self._visual_colors = np.tile(color, (self.num_envs, self.num_bodies, 1))
        elif self.cfg.randomization.randomize_color == 2:
            params = self.cfg.randomization.color_sampling_params
            color = params[0] + np.random.rand(self.num_envs, self.num_bodies, 3) * params[1]
            self._visual_colors = np.clip(color, 0, 1)
        else:
            raise ValueError(
                f"Invalid randomization of color category for asset: {self.cfg.randomization.randomize_color}."
            )

    def _process_rigid_shape_props(
        self, props: List[gymapi.RigidShapeProperties], env_id: int
    ) -> List[gymapi.RigidShapeProperties]:
        """Store/change/randomize the rigid shape properties of actor in each environment.

        Currently the function randomizes the following properties:
        - friction material on all rigid shapes (collision bodies)

        Note:
            This function is called during spawning of the asset into the environment.

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment index.

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # randomize friction material on all rigid bodies
        if self.cfg.randomization.randomize_friction:
            for index in range(len(props)):
                props[index].friction = self._friction_coeffs[env_id]
        if self.cfg.randomization.randomize_restitution:
            for index in range(len(props)):
                props[index].restitution = self._friction_coeffs[env_id]
        return props

    def _process_rigid_body_props(
        self, props: List[gymapi.RigidBodyProperties], env_id: int
    ) -> List[gymapi.RigidBodyProperties]:
        """Store/change/randomize the rigid body properties of actor in each environment.

        Currently the function randomizes the following properties:
        - adds mass on specified rigid body indices

        Note:
            This function is called during spawning of the asset into the environment.

        Args:
            props (List[gymapi.RigidBodyProperties]): Properties of each body of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidBodyProperties]]: Modified rigid body properties
        """
        # randomize added mass on specific rigid bodies
        if self.cfg.randomization.randomize_added_mass:
            body_indices = self.cfg.randomization.added_mass_rigid_body_indices
            for i, body_index in enumerate(body_indices):
                props[body_index].mass += self._added_masses[env_id, i]
        return props


# class CuboidAsset(Asset):
#     """Helper class for spawning a cuboid into simulation."""

#     def __init__(self, cfg: CuboidAssetCfg, num_envs: int, gym_iface: GymInterface):
#         super().__init__(cfg, num_envs, gym_iface)
#         # note: we reassign cfg here for PyLance to recognize the class object
#         self.cfg = cfg

#     def _retrieve_asset_instance(self, asset_options: gymapi.AssetOptions) -> gymapi.Asset:
#         return self.gym.create_box(self.sim, self.cfg.width, self.cfg.height, self.cfg.depth, asset_options)


# class SphereAsset(Asset):
#     """Helper class for spawning a sphere into simulation."""

#     def __init__(self, cfg: SphereAssetCfg, num_envs: int, gym_iface: GymInterface):
#         super().__init__(cfg, num_envs, gym_iface)
#         # note: we reassign cfg here for PyLance to recognize the class object
#         self.cfg = cfg

#     def _retrieve_asset_instance(self, asset_options: gymapi.AssetOptions) -> gymapi.Asset:
#         return self.gym.create_sphere(self.sim, self.cfg.radius, asset_options)


class FileAsset(Asset):
    """Helper class for spawning an asset from a file into simulation.

    Note: Supported formats include: URDF, MJCF and USD.
    """

    def __init__(self, cfg: FileAssetCfg, num_envs: int, gym_iface: GymInterface):
        super().__init__(cfg, num_envs, gym_iface)
        # note: we reassign cfg here for PyLance to recognize the class object
        self.cfg = cfg

    def _retrieve_asset_instance(self, asset_options: gymapi.AssetOptions) -> gymapi.Asset:
        # resolve paths
        asset_root = self.cfg.asset_root
        asset_file = self.cfg.asset_file
        asset_path = asset_root + asset_file
 
        # check file exists
        if not os.path.exists(asset_path):
            raise FileNotFoundError(f"Asset file does not exist: {asset_path}")

        return self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
