
from nav_gym.nav_legged_gym.common.sensors.sensors import Raycaster,SensorBase
from nav_gym.nav_legged_gym.envs.legged_nav_env_config import LeggedNavEnvCfg
from typing import Dict
class SensorManager:
    def __init__(self,env):
        self.sensor_cfg = env.cfg.sensors
        self.sensors_dict:Dict[str,SensorBase] = {}
        for raycaster_name, raycaster_cfg in self.sensor_cfg.raycasters_dict.items():
            if self.sensors_dict.__contains__(raycaster_name):
                print("[ERROR]raycaster name already exists: ", raycaster_name)
            else:
                self.sensors_dict[raycaster_name] = Raycaster(raycaster_cfg,env)
    def update(self):
        for sensor_name, sensor in self.sensors_dict.items():
            sensor.update() 
    def debug_vis(self):
        for sensor_name, sensor in self.sensors_dict.items():
            sensor.debug_vis() 
        
if __name__ == "__main__":
    print(LeggedNavEnvCfg.sensors.raycasters_dict.items())