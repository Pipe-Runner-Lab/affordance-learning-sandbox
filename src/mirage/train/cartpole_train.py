from omni.isaac.gym.vec_env import VecEnvBase

env = VecEnvBase(headless=True)

from ..tasks.cartpol_task import CartpoleTask

task = CartpoleTask(name="Cartpole")
env.set_task(task, backend="torch")

from stable_baselines3 import PPO

# create agent from stable baselines
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1000,
    batch_size=1000,
    n_epochs=20,
    learning_rate=0.001,
    gamma=0.99,
    device="cuda:0",
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=1.0,
    verbose=1,
    tensorboard_log="./cartpole_tensorboard",
)
model.learn(total_timesteps=100000)
model.save("ppo_cartpole")

env.close()
