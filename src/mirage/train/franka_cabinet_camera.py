# Import the skrl components to build the RL system
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed
from ..models.ppo import Policy, Value
from ..tasks.franka_cabinet_camera.config import TASK_CFG

# set the seed for reproducibility
set_seed(TASK_CFG["seed"])

# instance VecEnvBase and setup task
env = get_env_instance(headless=TASK_CFG["headless"])

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig  # noqa
from ..tasks.franka_cabinet_camera.task import FrankaCabinetCameraTask  # noqa

sim_config = SimConfig(TASK_CFG)
task = FrankaCabinetCameraTask(
    name="Franka Cabinet Camera", sim_config=sim_config, env=env
)
env.set_task(
    task=task,
    sim_params=sim_config.get_physics_params(),
    backend="torch",
    init_sim=True,
)

# wrap the environment
env = wrap_env(env, "omniverse-isaacgym")

device = env.device

# Instantiate a RandomMemory as rollout buffer (any memory
# can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {
    "policy": Policy(env.observation_space, env.action_space, device),
    "value": Value(env.observation_space, env.action_space, device),
}


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to
# see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 16
cfg_ppo["learning_epochs"] = 8
cfg_ppo["mini_batches"] = 8
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 5e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 2.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {
    "size": env.observation_space,
    "device": device,
}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 32 and 250 timesteps
# respectively
cfg_ppo["experiment"]["write_interval"] = 32
cfg_ppo["experiment"]["checkpoint_interval"] = 250

agent = PPO(
    models=models_ppo,
    memory=memory,
    cfg=cfg_ppo,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 5000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
