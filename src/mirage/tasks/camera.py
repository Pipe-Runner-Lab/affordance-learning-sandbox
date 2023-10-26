import math
from omni.usd import get_context
from omni.kit.viewport_legacy import get_viewport_interface
from omni.isaac.synthetic_utils import SyntheticDataHelper

from omni.isaac.core.utils.prims import create_prim, define_prim
from omni.isaac.core import World
from pxr import Gf
from omni.isaac.synthetic_utils import SyntheticDataHelper

sd_helper = SyntheticDataHelper()


class Camera:
    def __init__(
        self,
        id: str,
        width: int,
        height: int,
        fov,
        near,
        far,
        headless: bool = False,
        path: str = "/World",
    ):
        """

        Args:
            id: The id of the camera
            width: The horizontal image resolution in pixels
            height: The vertical image resolution in pixels
            fov: The field of view of the camera
            near: The near plane distance
            far: The far plane distance
        """

        self.id = id
        self._width = width
        self._height = height
        self.__fov = fov
        self.__near = near
        self.__far = far
        self.__aspect = (
            self._width / self._height
        )  # TODO: see why it is not 62 degrees the horizontalAperture !!!!
        self._view_matrix = None

        self.camera_prim_path = f"{path}/{id}"
        fov_horizontal = self.__aspect * fov
        focal_length = 1.88
        attributes = {
            "horizontalAperture": 2
            * focal_length
            * math.tan(fov_horizontal * math.pi / 180 / 2),
            "verticalAperture": 2
            * focal_length
            * math.tan(fov * math.pi / 180 / 2),
            "focalLength": focal_length,
            "clippingRange": (self.__near, self.__far),
        }
        create_prim(
            prim_path=self.camera_prim_path,
            prim_type="Camera",
            attributes=attributes,
        )  # , "clippingRange": (self.__near, self.__far)}) #, "clippingRange": (self.__near, self.__far)}) #, attributes={"width": self._width})
        self.stage = get_context().get_stage()
        self.camera_prim = self.stage.GetPrimAtPath(self.camera_prim_path)

        # Set as current camera
        if headless:
            viewport_interface = get_viewport_interface()
            self.viewport = viewport_interface.get_viewport_window()
        else:
            viewport_handle = get_viewport_interface().create_instance()
            list_viewports = get_viewport_interface().get_instance_list()
            new_viewport_name = (
                get_viewport_interface().get_viewport_window_name(
                    viewport_handle
                )
            )
            self.viewport = ().get_viewport_window(viewport_handle)
            window_width = 200
            window_height = 200
            self.viewport.set_window_size(window_width, window_height)
            self.viewport.set_window_pos(
                800, window_height * (len(list_viewports) - 2)
            )

        self.viewport.set_active_camera(self.camera_prim_path)
        self.viewport.set_texture_resolution(self._width, self._height)

    def get_image(self):
        # Get ground truths
        gt = sd_helper.get_groundtruth(
            [
                "rgb",
                # "depthLinear",
                "depth",
                # "boundingBox2DTight",
                # "boundingBox2DLoose",
                "instanceSegmentation",
                # "semanticSegmentation",
                # "boundingBox3D",
                # "camera",
                # "pose"
            ],
            self.viewport,
        )

        print("Camera params", sd_helper.get_camera_params(self.viewport))

        segmentation_mask = gt["instanceSegmentation"]
        rgb = gt["rgb"]
        depth = gt["depth"]
        return rgb, depth, segmentation_mask

    def get_pose(self):
        transform_matrix = sd_helper.get_camera_params(self.viewport)["pose"]
        return transform_matrix

    def set_prim_pose(self, position, orientation):
        properties = self.camera_prim.GetPropertyNames()
        if "xformOp:translate" in properties:
            translate_attr = self.camera_prim.GetAttribute("xformOp:translate")
            translate_attr.Set(Gf.Vec3d(position))
        if "xformOp:orient" in properties:
            orientation_attr = self.camera_prim.GetAttribute("xformOp:orient")
            orientation_attr.Set(
                Gf.Quatd(
                    orientation[0],
                    orientation[1],
                    orientation[2],
                    orientation[3],
                )
            )


if __name__ == "__main__":
    world = World()
    world.scene.add_default_ground_plane()
    path_to = "/World/Scene"
    define_prim(path_to, "Xform")
    stage = get_context().get_stage()
    world.reset()
    camera = Camera(
        id="my_camera",
        width=224,
        height=171,
        fov=45,
        near=0.10,
        far=4,
        path="/World/Scene",
    )
    camera.set_prim_pose(position=[0, 0, 1], orientation=[0, 0, 0, 1])
    print(camera.get_image())
