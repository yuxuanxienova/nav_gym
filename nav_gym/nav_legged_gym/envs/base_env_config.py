# python
from dataclasses import MISSING

# legged-gym
from nav_gym.nav_legged_gym.utils.config_utils import configclass
from nav_gym.nav_legged_gym.common.gym_interface import GymInterfaceCfg


@configclass
class EnvCfg:
    """Common configuration for environment."""

    num_envs: int = MISSING
    """Number of environment instances."""

    num_actions: int = MISSING  # joint positions, velocities or torques
    """The size of action space for the defined environment MDP.

    TODO (@mmittal): Generalize with action procesing controllers.
    """

    episode_length_s: float = MISSING
    """Episode length in seconds."""

    send_timeouts: bool = True  # send time out information to the algorithm
    """Whether to send episode time-out information (added as part of infos)."""

    enable_debug_vis: bool = False


@configclass
class ControlCfg:
    """Control configuration for stepping the environment."""

    decimation: int = 1
    """Number of times to apply control action, i.e. number of simulation time-steps per policy time-step."""

    action_clipping: float = 100.0
    """Clipping of actions provided to the environment."""

    action_scale: float = 1.0
    """Scaling of input actions provided to the environment."""


@configclass
class BaseEnvCfg:
    """Basic configuration required by all environments to provide."""

    # initialize the configurations.
    # -- common
    env: EnvCfg = EnvCfg()
    gym: GymInterfaceCfg = GymInterfaceCfg()
    # -- action processing
    control: ControlCfg = ControlCfg()
