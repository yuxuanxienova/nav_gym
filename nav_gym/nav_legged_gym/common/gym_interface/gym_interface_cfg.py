# python
from typing import Tuple

# legged-gym
from nav_gym.nav_legged_gym.utils.config_utils import configclass


@configclass
class PhysxCfg:
    """Parameters related to PhysX physics engine.

    Reference: :cls:`isaacgym.gymapi.PhysXParams`
    """

    use_gpu: bool = True
    """Use PhysX GPU."""

    num_threads: int = 10
    """Number of CPU threads used by PhysX."""

    num_subscenes: int = 0
    """Number of subscenes for multithreaded simulation."""

    solver_type: int = 1
    """Type of solver to use. 0: PGS, 1: TGS."""

    num_position_iterations: int = 4
    """PhysX solver position iterations count. Range [1,255]"""

    num_velocity_iterations: int = 0
    """PhysX solver velocity iterations count. Range [1,255]"""

    contact_collection: int = 2
    """Contact collection mode. 0: never, 1: last sub-step, 2: all sub-steps."""

    contact_offset: float = 0.01
    """Shapes whose distance is less than the sum of their contactOffset values will generate contacts."""

    rest_offset = 0.0  # [m]
    """Two shapes will come to rest at a distance equal to the sum of their restOffset values."""

    bounce_threshold_velocity: float = 0.5  # [m/s]
    """A contact with a relative velocity below this will not bounce."""

    max_depenetration_velocity: float = 1.0
    """The maximum velocity permitted to be introduced by the solver to correct for penetrations in contacts."""

    max_gpu_contact_pairs: int = 2**23  # 2**24 -> needed for 8000 envs and more
    """Maximum number of contact pairs."""

    default_buffer_size_multiplier: int = 5
    """Default buffer size multiplier."""


@configclass
class SimParamsCfg:
    """Gym simulation parameters.

    Reference: :cls:`isaacgym.gymapi.SimParams`
    """

    dt: float = 0.005
    substeps: int = 1
    up_axis: str = "UP_AXIS_Z"  # "UP_AXIS_Y", "UP_AXIS_Z"
    gravity: tuple = (0.0, 0.0, -9.81)
    use_gpu_pipeline: bool = True

    physx: PhysxCfg = PhysxCfg()
    """PhysX specific simulation parameters."""


@configclass
class ViewerCfg:
    """Configuration for the viewer camera."""

    # gymapi.DEFAULT_VIEWER_WIDTH= 1600
    # gymapi.DEFAULT_VIEWER_HEIGHT= 900

    ref_env: int = 0
    """Reference environment w.r.t. set camera position."""
    eye: Tuple[float, float, float] = (5.0, 5.0, 5.0)
    """Location of camera eye (in m)."""
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Location of target point to look at (in m)."""


@configclass
class GymInterfaceCfg:
    """Parameters for interface class to IsaacGym."""

    physics_engine: str = "physx"  # Choose from physx or flex
    sim_device: str = "cuda:0"#"cpu" or "cuda:0"   Device of simulation
    graphics_device_id: int = 0  # Device of visualization
    headless: bool = False  # Do without visual window

    sim_params: SimParamsCfg = SimParamsCfg()
    viewer: ViewerCfg = ViewerCfg()

if __name__ == "__main__":
    gyminterfacecfg = GymInterfaceCfg(graphics_device_id=0)
    gyminterfacecfg = GymInterfaceCfg(SimParamsCfg(dt=0.01))


