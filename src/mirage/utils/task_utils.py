from typing import Dict
from mirage.tasks.franka_block_task import FrankaBlockTask
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames


def initialize_task(cfg_dict: Dict, env: VecEnvRLGames, init_sim: bool = True):
    sim_config = SimConfig(cfg_dict)

    cfg = sim_config.config
    task = FrankaBlockTask(
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend="torch",
        init_sim=init_sim,
    )

    return task
