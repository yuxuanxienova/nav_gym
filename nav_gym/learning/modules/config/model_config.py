from typing import List, Tuple, Union, Optional
from legged_gym.utils.config_utils import configclass
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
class GaussianDistributionCfg:
    model_class: str = "Gaussian"
    init_std: float = 1.0