# python
from dataclasses import MISSING

# legged-gym
from nav_gym.nav_legged_gym.utils.config_utils import configclass


@configclass
class ActuatorCfg:
    cls_name: str = "Actuator"
    control_type = "explicit"  # 'explicit': for
    command_type = "P"  # 'P': position, 'V': velocity, 'T': torque
    enable_torque_sensor: bool = False
    motor_torque_limit: float = None  # Nm, enforced by PhysX
    motor_velocity_limit: float = None  # rad/s, not enforced by the ideal Actuator
    gear_ratio = 1.0  # used to compute dof quantities from motor quantities


@configclass
class DCMotorCfg(ActuatorCfg):
    cls_name = "DCMotor"
    peak_motor_torque: float = (
        10.0  # Nm, motor_torque_limit = min(peak_motor_torque*(1 - dof_vel/motor_velocity_limit), torque_limit)
    )


@configclass
class VariableGearRatioDCMotorCfg(ActuatorCfg):
    cls_name = "VariableGearRatioDCMotor"

    peak_motor_torque = 20.0
    motor_velocity_limit = 100.0
    motor_torque_limit = 10.0
    gear_ratio_function = "lambda x: torch.square(x -1.)"  # example function


@configclass
class ActuatorNetMLPCfg(DCMotorCfg):
    cls_name = "ActuatorNetMLP"

    network_file = MISSING
    vel_scale = 0.2
    pos_scale = 5.0
    torque_scale = 60.0
    input_idx = [0, 2, 4]
    # input_idx = [0, 4, 8]


@configclass
class ActuatorNetLSTMCfg(DCMotorCfg):
    cls_name = "ActuatorNetLSTM"

    network_file: str = MISSING


""" Instances """
import os
anymal_d_actuator_cfg = ActuatorNetMLPCfg(
    network_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "resources") + "/actuator_model/anymal_d_actuator.jit",
    gear_ratio=1.0,
    peak_motor_torque=140.0,
    motor_velocity_limit=8.5,
    motor_torque_limit=89.0,
)
# 48V
barry_hip_actuator = DCMotorCfg(
    gear_ratio=5.6, motor_torque_limit=140 / 5.6, peak_motor_torque=152.49 / 5.6, motor_velocity_limit=13.97 * 5.6
)

# 48V
barry_knee_actuator = VariableGearRatioDCMotorCfg(
    motor_torque_limit=50.0,
    peak_motor_torque=27.23,  # lower than the torque limit at 48V
    motor_velocity_limit=39.96,
    gear_ratio_function="lambda x:  9 * torch.cos(-x - 1.74533)",
)

# wheels with simple velocity control
wheel_actuator = ActuatorCfg(
    control_type="implicit",
    command_type="V",
    enable_torque_sensor=True,
    motor_torque_limit=30.0,
    motor_velocity_limit=50.0,
    gear_ratio=1.0,
)

# 48V
baboon_actuator = DCMotorCfg(
    gear_ratio=1.0,
    motor_torque_limit=27.0,
    peak_motor_torque=60.0,
    motor_velocity_limit=15.5,
)

# 48V
coyote_actuator = DCMotorCfg(
    gear_ratio=1.0,
    motor_torque_limit=14.0,
    peak_motor_torque=30.0,
    motor_velocity_limit=40.0,
)

if __name__ == "__main__":
    # check asset file path
    import os

    cfg = anymal_d_actuator_cfg
    asset_path = cfg.network_file
    if not os.path.exists(asset_path):
        raise FileNotFoundError(f"Asset file does not exist: {asset_path}")
    print("Asset file path is correct.")