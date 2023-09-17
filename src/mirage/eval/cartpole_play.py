# create isaac environment
from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

# create task and register task
from ..tasks.cartpol_task import CartpoleTask
task = CartpoleTask(name="Cartpole")
env.set_task(task, backend="torch")

# import stable baselines
from stable_baselines3 import PPO

# Run inference on the trained policy
model = PPO.load("ppo_cartpole")
env._world.reset()
obs = env.reset()
while env._simulation_app.is_running():
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

env.close()