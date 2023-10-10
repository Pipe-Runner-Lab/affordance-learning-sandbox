import torch
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.robots.articulations.franka import Franka as Robot
from omniisaacgymenvs.robots.articulations.views.franka_view import (
    FrankaView as RobotView,
)
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.scenes.scene import Scene


def spawn_robot(task: RLTask):
    """Spawn franka robot in the scene.
    Note: The view function should only be called after
    super().set_up_scene() is called since set_up_scene is responsible for
    cloning the environments.

    Args:
        task (RLTask): The task object

    Returns:
        get_robot_view: A function that takes in a scene object
        and returns a robot view
        (Used for actuating and querying all the robots).
        Note: The robot, its hand, left and right fingers are also added to
        the scene for articulation.
    """

    robot = Robot(
        prim_path=task.default_zero_env_path + "/franka",
        translation=torch.tensor([0.0, 0.0, 0.0]),
        orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        name="robot",
    )

    task._sim_config.apply_articulation_settings(
        "robot",  # this parameter is useless if config is provided
        get_prim_at_path(robot.prim_path),
        task._sim_config.parse_actor_config("robot"),
    )

    def get_robot_view(scene: Scene):
        robots = RobotView(
            prim_paths_expr="/World/envs/.*/franka", name="robot_view"
        )
        scene.add(robots)
        scene.add(robots._hands)
        scene.add(robots._lfingers)
        scene.add(robots._rfingers)
        return robots

    return get_robot_view


def spawn_target(task: RLTask):
    box_color = torch.tensor([0.2, 0.4, 0.6])
    box_pos = torch.tensor([0.3, 0.3, 0.0])

    box_size = 0.08

    box = DynamicCuboid(
        prim_path=task.default_zero_env_path + "/target",
        name="target",
        color=box_color,
        size=box_size,
        density=100.0,
        translation=box_pos,
    )

    task._sim_config.apply_articulation_settings(
        "target",
        get_prim_at_path(box.prim_path),
        task._sim_config.parse_actor_config("target"),
    )

    def get_target_view(scene: Scene):
        targets = RigidPrimView(
            prim_paths_expr="/World/envs/.*/target", name="target_view"
        )
        scene.add(targets)
        return targets

    return get_target_view
