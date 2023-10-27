from ..tasks.franka_cabinet_camera.config import TASK_CFG
import torch
from omni.isaac.gym.vec_env import VecEnvBase


def run_env():
    env = VecEnvBase(headless=True)

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    from ..tasks.franka_cabinet_camera.task import FrankaCabinetCameraTask

    sim_config = SimConfig(TASK_CFG)
    task = FrankaCabinetCameraTask(
        name="Debug", sim_config=sim_config, env=env
    )
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
            (task._num_envs, task.num_franka_dofs),
            dtype=torch.float,
            device=task._device,
        )
        obs, rewards, dones, info = env.step(action)

    env.close()


run_env()
