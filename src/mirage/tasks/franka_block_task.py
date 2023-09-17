from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.objects import DynamicCuboid
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from omniisaacgymenvs.robots.articulations.views.cabinet_view import (
    CabinetView,
)
from omni.isaac.core.prims import RigidPrimView
import numpy as np
import torch
import math
from omni.isaac.cloner import Cloner


class FrankaBlockTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None):
        self.sim_config = sim_config

        self.dt = 1 / 60.0

        self._num_observations = 23
        self._num_actions = 9

        self._sim_config = sim_config
        self._task_cfg = sim_config.task_config
        self.num_props = self._task_cfg["env"]["numProps"]

        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene, replicate_physics=True) -> None:
        self.get_franka()
        if self.num_props > 0:
            self.get_props()

        super().set_up_scene(scene)

        self._frankas = FrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="franka_view"
        )
        self._cabinets = CabinetView(
            prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view"
        )

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*",
                name="prop_view",
                reset_xform_properties=False,
            )
            scene.add(self._props)

        self.init_data()

    def get_franka(self):
        franka = Franka(
            prim_path=self.default_zero_env_path + "/franka", name="franka"
        )
        self._sim_config.apply_articulation_settings(
            "franka",
            get_prim_at_path(franka.prim_path),
            self._sim_config.parse_actor_config("franka"),
        )

    def get_props(self) -> None:
        prop_cloner = Cloner()
        drawer_pos = torch.tensor([0.0515, 0.0, 0.7172])
        prop_color = torch.tensor([0.2, 0.4, 0.6])

        props_per_row = int(math.ceil(math.sqrt(self.num_props)))
        prop_size = 0.08
        prop_spacing = 0.09
        xmin = -0.5 * prop_spacing * (props_per_row - 1)
        zmin = -0.5 * prop_spacing * (props_per_row - 1)
        prop_count = 0

        prop_pos = []
        for j in range(props_per_row):
            prop_up = zmin + j * prop_spacing
            for k in range(props_per_row):
                if prop_count >= self.num_props:
                    break
                propx = xmin + k * prop_spacing
                prop_pos.append([propx, prop_up, 0.0])

                prop_count += 1

        prop = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/prop/prop_0",
            name="prop",
            color=prop_color,
            size=prop_size,
            density=100.0,
        )

        self._sim_config.apply_articulation_settings(
            "prop",
            get_prim_at_path(prop.prim_path),
            self._sim_config.parse_actor_config("prop"),
        )

        prop_paths = [
            f"{self.default_zero_env_path}/prop/prop_{j}"
            for j in range(self.num_props)
        ]
        prop_cloner.clone(
            source_prim_path=self.default_zero_env_path + "/prop/prop_0",
            prim_paths=prop_paths,
            positions=np.array(prop_pos) + drawer_pos.numpy(),
            replicate_physics=False,
        )

    def init_data(self):
        pass

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def post_physics_step(self, step_size: float) -> None:
        pass
