from omni.isaac.sensor import RotatingLidarPhysX


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
