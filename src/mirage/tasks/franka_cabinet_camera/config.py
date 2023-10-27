from ..franka_cabinet.config import TASK_CFG

TASK_CFG["headless"] = True

TASK_CFG["task"]["env"]["numEnvs"] = 1
TASK_CFG["task"]["env"]["cameraWidth"] = 16
TASK_CFG["task"]["env"]["cameraHeight"] = 16
TASK_CFG["task"]["env"]["exportImages"] = True
TASK_CFG["task"]["env"]["envSpacing"] = 20

TASK_CFG["task"]["sim"]["rendering_dt"] = 0.0166  # 1/60 half of physics step
TASK_CFG["task"]["sim"]["enable_cameras"] = True
TASK_CFG["task"]["sim"]["add_ground_plane"] = True
TASK_CFG["task"]["sim"]["add_distant_light"] = True
