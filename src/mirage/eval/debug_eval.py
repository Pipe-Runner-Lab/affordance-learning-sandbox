import torch
from ..tasks.franka_cabinet_lidar.config import TASK_CFG
from omni.isaac.gym.vec_env import VecEnvBase

TASK_CFG["headless"] = False
TASK_CFG["task"]["env"]["numEnvs"] = 1

env = VecEnvBase(headless=TASK_CFG["headless"])

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig  # noqa
from ..tasks.franka_cabinet_lidar.task import FrankaCabinetLidarTask  # noqa

sim_config = SimConfig(TASK_CFG)
task = FrankaCabinetLidarTask(name="Debug", sim_config=sim_config, env=env)
env.set_task(
    task=task,
    sim_params=sim_config.get_physics_params(),
    backend="torch",
    init_sim=True,
)

env._world.reset()
obs = env.reset()
while env._simulation_app.is_running():
    # action, _states = model.predict(obs)
    action = torch.zeros(
        (task._num_envs, task._num_actions),
        dtype=torch.float,
        device=task._device,
    )
    obs, rewards, dones, info = env.step(action)

env.close()
