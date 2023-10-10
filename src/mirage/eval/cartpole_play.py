# create isaac environment
from omni.isaac.gym.vec_env import VecEnvBase
from stable_baselines3 import PPO

env = VecEnvBase(headless=False)

from ..tasks.cartpol_task import CartpoleTask

task = CartpoleTask(name="Cartpole")
env.set_task(task, backend="torch")

# Run inference on the trained policy
model = PPO.load("ppo_cartpole")
env._world.reset()
obs = env.reset()
while env._simulation_app.is_running():
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

env.close()
