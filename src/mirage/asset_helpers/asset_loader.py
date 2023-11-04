import torch
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.robots.articulations.franka import Franka as Robot
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
from omniisaacgymenvs.robots.articulations.views.franka_view import (
    FrankaView as RobotView,
)
from .custom_view import CabinetView
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.scenes.scene import Scene


def spawn_robot(task: RLTask, usd_path=None):
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
        translation=torch.tensor([1.0, 0.0, 0.0]),
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        name="robot",
        usd_path=usd_path,
    )

    task._sim_config.apply_articulation_settings(
        "robot",  # this parameter is useless if config is provided
        get_prim_at_path(robot.prim_path),
        task._sim_config.parse_actor_config("robot"),
    )

    def get_robot_view(scene: Scene):
        view_object_name = "robot_view"

        if scene.object_exists(view_object_name):
            scene.remove_object(view_object_name, registry_only=True)

        # have to use "franka" here because lfingers and rfingers are children
        # of franka
        robots = RobotView(
            prim_paths_expr="/World/envs/.*/franka", name=view_object_name
        )
        scene.add(robots)
        scene.add(robots._hands)
        scene.add(robots._lfingers)
        scene.add(robots._rfingers)
        return robots

    return get_robot_view


def spawn_cabinet(task: RLTask):
    cabinet = Cabinet(
        task.default_zero_env_path + "/cabinet",
        translation=torch.tensor([0.0, 0.0, 0.4]),
        orientation=torch.tensor([0.1, 0.0, 0.0, 0.0]),
        name="cabinet",
    )
    task._sim_config.apply_articulation_settings(
        "cabinet",
        get_prim_at_path(cabinet.prim_path),
        task._sim_config.parse_actor_config("cabinet"),
    )

    def get_cabinet_view(scene: Scene):
        view_object_name = "cabinet_view"

        if scene.object_exists(view_object_name):
            scene.remove_object(view_object_name, registry_only=True)

        cabinet = CabinetView(
            prim_paths_expr="/World/envs/.*/cabinet", name=view_object_name
        )
        scene.add(cabinet)
        scene.add(cabinet._top_drawers)
        scene.add(cabinet._bottom_drawers)
        return cabinet

    return get_cabinet_view


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
        view_object_name = "target_view"

        if scene.object_exists(view_object_name):
            scene.remove_object(view_object_name, registry_only=True)

        targets = RigidPrimView(
            prim_paths_expr="/World/envs/.*/target", name=view_object_name
        )
        scene.add(targets)
        return targets

    return get_target_view
