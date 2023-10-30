from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.torch import wrap_env
from ..tasks.franka_cabinet.config import TASK_CFG
from ..models.ppo import Policy

TASK_CFG["task"]["env"]["numEnvs"] = 64
TASK_CFG["headless"] = False

# instance VecEnvBase and setup task
env = get_env_instance(
    headless=TASK_CFG["headless"]
)  # both this and config needs to be headless

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig  # noqa
from ..tasks.franka_cabinet.task import FrankaCabinetTask  # noqa: E402

sim_config = SimConfig(TASK_CFG)
task = FrankaCabinetTask(
    name="AffordanceBlockPickPlace", sim_config=sim_config, env=env
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

models_ppo = {
    "policy": Policy(env.observation_space, env.action_space, device),
}

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {
    "size": env.observation_space,
    "device": device,
}
# logging to TensorBoard each 32 timesteps an ignore checkpoints
cfg_ppo["experiment"]["write_interval"] = 32
cfg_ppo["experiment"]["checkpoint_interval"] = 0

agent = PPO(
    models=models_ppo,
    memory=None,
    cfg=cfg_ppo,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# load checkpoints
agent.load("runs/23-10-27_13-42-44-236910_PPO/checkpoints/best_agent.pt")

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 5000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start evaluation
trainer.eval()