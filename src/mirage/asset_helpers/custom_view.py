from omni.isaac.sensor import RotatingLidarPhysX
from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class CabinetView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "CabinetView",
    ) -> None:
        """[summary]"""

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False,
        )

        self._top_drawers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/cabinet/drawer_top",
            name="top_drawers_view",
            reset_xform_properties=False,
        )
        self._bottom_drawers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/cabinet/drawer_bottom",
            name="bottom_drawers_view",
            reset_xform_properties=False,
        )


class LidarView(RotatingLidarPhysX):
    def __init__(self, prim_paths: str, name: str = "lidar") -> None:
        super().__init__(
            prim_paths,
            name,
            rotation_frequency=0,
            fov=(160, 30),
            resolution=(0.4, 4.0),
            valid_range=(0.2, 2.3),
        )
        self.add_depth_data_to_frame()
        self.add_point_cloud_data_to_frame()
        self.add_semantics_data_to_frame()
        self.enable_visualization(
            high_lod=False, draw_points=False, draw_lines=True
        )
        self.initialize()
