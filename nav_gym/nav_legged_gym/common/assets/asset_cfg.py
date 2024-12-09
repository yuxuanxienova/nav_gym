from dataclasses import MISSING
from typing import Tuple
import os
from nav_gym.nav_legged_gym.utils.config_utils import configclass
from nav_gym import NAV_GYM_ROOT_DIR

__all__ = ["AssetCfg", "CuboidAssetCfg", "SphereAssetCfg", "FileAssetCfg"]


@configclass
class AssetCfg:
    """Common configuration for importing any asset into simulation.

    For available asset options: check :cls:`isaacgym.gymapi.AssetOptions`.
    """

    cls_name: str = MISSING
    """Name of the class to create instance from. Used by the environment when parsing configuration."""

    asset_name: str = "asset"
    """Name of the asset. Used in the GUI."""

    self_collisions: bool = True
    """Self collision bitwise filter: 1 to disable, 0 to enable."""

    segmentation_id: int = 0
    """Segmentation id for the asset (used by images only)."""

    ##
    # Asset options.
    ##

    default_dof_drive_mode: int = 3
    """See `GymDofDriveModeFlags` (0: none, 1: position target, 2: velocity target, 3: effort)."""

    fix_base_link: bool = False
    """Fix the root of the asset."""

    disable_gravity: bool = False
    """Disable gravity on all bodies of the asset."""

    collapse_fixed_joints: bool = True
    """Merge bodies connected by fixed joints. Specific fixed joints can be kept by adding `<... dont_collapse="true">`."""

    replace_cylinder_with_capsule: bool = True
    """Replace collision cylinders with capsules, leads to faster/more stable simulation."""

    flip_visual_attachments: bool = True
    """Some .obj meshes must be flipped from y-up to z-up."""

    density: float = MISSING
    """Default density used for bodies that don't have a pre-specified mass/inertia."""

    angular_damping: float = 0.0
    """Angular velocity damping for rigid bodies."""

    linear_damping: float = 0.0
    """Linear velocity damping for rigid bodies."""

    max_angular_velocity: float = 1000.0
    """Maximum angular velocity for rigid bodies."""

    max_linear_velocity: float = 1000.0
    """Maximum linear velocity for rigid bodies."""

    armature: float = 0.0
    """Added to the diagonal elements of inertia tensors for all of the asset's rigid bodies/links. Could improve simulation stability."""

    thickness: float = 0.01
    """Thickness of the collision shapes. Sets how far objects should come to rest from the surface of this body."""

    ##
    # Initial state of robot.
    ##

    @configclass
    class InitState:
        """Initial state of the asset."""

        pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # x,y,z [m]
        rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # x,y,z,w [quat]
        lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x,y,z [m/s]
        ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x,y,z [rad/s]

    init_state: InitState = InitState()

    ##
    # Randomization settings.
    ##

    @configclass
    class Randomization:
        """Property randomization of asset.

        If enabled, properties are sampled from uniform distribution within specified range.

        Note:
            The way friction is combined between two assets will change:
                - Preview 3: MULTIPLY
                - Preview 4: AVERAGE
        """

        # flags for different randomization
        randomize_friction: bool = False
        randomize_restitution: bool = False
        randomize_added_mass: bool = False
        randomize_color: int = 0  # 0: "none", 1: "fixed", 2: "random"

        # properties specifications
        # -- friction
        friction_buckets: int = 64
        """Number of physics material to create in the simulation."""
        friction_range: Tuple[float, float] = (0.0, 1.0)
        """The range of the uniform distribution for sampling coefficients."""
        # -- restitution
        restitution_buckets: int = 32
        """Number of physics material to create in the simulation."""
        restitution_range: Tuple[float, float] = (0.0, 0.5)
        """The range of the uniform distribution for sampling coefficients."""
        # -- rigid body mass
        added_mass_range: Tuple[float, float] = (-1.0, 1.0)
        """The range of the uniform distribution for sampling masses."""
        added_mass_rigid_body_indices: Tuple[int, ...] = (0,)
        """Rigid body indices at which to add mass."""
        # -- color
        color_fixed = (0.0, 0.0, 0.0)
        """Fixed color for all bodies in actor."""
        color_sampling_params = (0.0, 1.0)
        """Parametric sampling of color: :math:`\alpha + \beta * Uniform(0, 1)`."""

    randomization: Randomization = Randomization()


# @configclass
# class CuboidAssetCfg(AssetCfg):
#     """Configuration for spawning a cuboid into simulation."""

#     cls_name: str = "CuboidAsset"

#     width: float = MISSING  # x-axis
#     height: float = MISSING  # y-xis
#     depth: float = MISSING  # z-axis


# @configclass
# class SphereAssetCfg(AssetCfg):
#     """Configuration for spawning a sphere into simulation."""

#     cls_name: str = "CuboidAsset"

#     radius: float = MISSING  # radius (m)


@configclass
class FileAssetCfg(AssetCfg):
    """Configuration for spawning an asset from a file into simulation."""

    cls_name: str = "FileAsset"
    # asset_root=os.path.join(NAV_GYM_ROOT_DIR,"resources")
    asset_root = MISSING
    asset_file = MISSING 

    density: float = 0.001  # Avoid adding weight to bodies without inertial properties.




