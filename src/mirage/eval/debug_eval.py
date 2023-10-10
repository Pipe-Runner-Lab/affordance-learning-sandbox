# this import also registers important hydra utils
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import OmegaConf
import torch

from omni.isaac.gym.vec_env import VecEnvBase
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict

import hydra
from omegaconf import DictConfig


# from stable_baselines3 import PPO
@hydra.main(config_name="config", config_path="../../../configs/experiments")
def run_env(cfg: DictConfig):
    env = VecEnvBase(headless=False)

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    from ..tasks.debug_task import DebugTask

    cfg_dict = omegaconf_to_dict(cfg)

    sim_config = SimConfig(cfg_dict)
    task = DebugTask(name="Debug", sim_config=sim_config, env=env)
    env.set_task(task, backend="torch")

    env._world.reset()
    obs = env.reset()
    while env._simulation_app.is_running():
        # action, _states = model.predict(obs)
        action = torch.zeros(
            (task._num_envs, task.num_franka_dofs),
            dtype=torch.float,
            device=task._device,
        )
        obs, rewards, dones, info = env.step(action)

    env.close()


run_env()
