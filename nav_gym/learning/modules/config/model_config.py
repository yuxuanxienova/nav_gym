from typing import List, Tuple, Union, Optional
from nav_gym.nav_legged_gym.utils.config_utils import configclass
#-------------Model Configs----------------
@configclass
class ArchitectureCfg:
    model_class: str = "TeacherModelBase"
    activation_fn: str = "elu"
    obs_names_critic: Optional[List[str]] = None  # TODO: for asymmetric critic
    scan_encoder_shape: List[int] = [128]
    scan_latent_size: int = 64
    priv_encoder_shape: Optional[List[int]] = None
    priv_latent_size: int = 16
    output_mlp_size: List[int] = [256, 128, 64]

@configclass
class FLDLocomotionActorCriticCfg:
    model_class: str = "FLDLocomotionActorCritic"
    activation_fn: str = "elu"
    obs_names_critic: Optional[List[str]] = None  # TODO: for asymmetric critic
    scan_encoder_shape: List[int] = [128]
    scan_latent_size: int = 64
    priv_encoder_shape: Optional[List[int]] = None
    priv_latent_size: int = 16
    output_mlp_size: List[int] = [256, 128, 64]

@configclass
class TeacherArchCfg:
    # model_class: str = "SimpleNavPolicy"
    model_class: str = "NavPolicyWithMemory"
    activation_fn: str = "elu"

    scan_cnn_channels: List[int] = [4, 4]
    scan_cnn_fc_shape: List[int] = [512, 256]
    scan_latent_size: int = 128

    # 1D conv for short history
    history_channels: List[int] = [8, 16]
    history_kernel_size: int = 3
    history_fc_shape: List[int] = [64]
    history_latent_size: int = 32

    output_mlp_size: List[int] = [256, 128]

    # for memory observation
    pointnet_channels: List[int] = [16, 32]
    pointnet_fc_shape: List[int] = [32, 32]
    aggregator_latent_size: int = 32
#--------------Distribution Configs----------------
@configclass
class GaussianDistributionCfg:
    model_class: str = "Gaussian"
    init_std: float = 1.0

@configclass
class betaActionDistCfg:
    model_class: str = "BetaDistribution"
    scale: float = 3.0  # initial scale to alpha + beta
    num_logits: int = 6  # SHOULD BE 2 x action dim