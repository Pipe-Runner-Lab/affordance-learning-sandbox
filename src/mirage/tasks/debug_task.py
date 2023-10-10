from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse
from omni.isaac.core.utils.torch.rotations import tensor_clamp
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrimView
from pxr import UsdGeom
import torch


class DebugTask(RLTask):
    def __init__(
        self,
        name: str,
        sim_config: SimConfig,
        env: VecEnvRLGames,
        offset: bool = None,
    ):
        self.dt = 1 / 60.0

        self._num_observations = 23
        self._num_actions = 9  # 7 franka dofs + 2 gripper dofs

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_envs = 2
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]

        RLTask.__init__(self, name, env)

    def set_up_scene(
        self, scene: Scene, replicate_physics: bool = True
    ) -> None:
        # set default camera viewport position and target
        self.set_initial_camera_params()

        # add default ground plane
        scene.add_default_ground_plane()

        # add franka and props to stage
        self.__add_franka_to_stage()
        self.__add_box_to_stage()

        """
            * INFO: Must be called after all objects are added to the scene
            * and before getting articulation views
        """
        super().set_up_scene(scene, replicate_physics)

        self.__frankas = FrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="franka_view"
        )

        scene.add(self.__frankas)
        scene.add(self.__frankas._hands)
        scene.add(self.__frankas._lfingers)
        scene.add(self.__frankas._rfingers)

        self.__boxes = RigidPrimView(
            prim_paths_expr="/World/envs/.*/box",
            name="box_view",
            reset_xform_properties=False,
        )
        scene.add(self.__boxes)

        self.init_data()

    # TODO: Read
    def init_data(self) -> None:
        # Note: We only compute for one env in local coordinates, and copy it
        # for _num_envs

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(  # pylint: disable=no-member
                stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")
            ),
            self._device,
        )
        lfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(  # pylint: disable=no-member
                stage.GetPrimAtPath(
                    "/World/envs/env_0/franka/panda_leftfinger"
                )
            ),
            self._device,
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(  # pylint: disable=no-member
                stage.GetPrimAtPath(
                    "/World/envs/env_0/franka/panda_rightfinger"
                )
            ),
            self._device,
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(
            hand_pose[3:7], hand_pose[0:3]
        )

        # Grasp pose includes both hand and finger pose
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
            hand_pose_inv_rot,
            hand_pose_inv_pos,
            finger_pose[3:7],
            finger_pose[0:3],
        )
        franka_local_pose_pos += torch.tensor(
            [0, 0.04, 0], device=self._device
        )

        # repeat for all envs
        self.__franka_local_grasp_pos = franka_local_pose_pos.repeat(
            (self._num_envs, 1)
        )
        self.__franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat(
            (self._num_envs, 1)
        )

        self.__gripper_forward_axis = torch.tensor(
            [0, 0, 1], device=self._device, dtype=torch.float
        ).repeat((self._num_envs, 1))
        self.__gripper_up_axis = torch.tensor(
            [0, 1, 0], device=self._device, dtype=torch.float
        ).repeat((self._num_envs, 1))

        self.__franka_default_dof_pos = torch.tensor(
            [
                1.157,
                -1.066,
                -0.155,
                -2.239,
                -1.841,
                1.003,
                0.469,
                0.035,
                0.035,
            ],
            device=self._device,
        )

        self.__actions = torch.zeros(
            (self._num_envs, self.num_actions), device=self._device
        )

    # TODO: Read
    def post_reset(self):
        """
        * INFO: This function is called after the scene has loaded and the
        * simulation is about to start. Used for getting indices of joints and
        * initial reset
        """

        self.num_franka_dofs = self.__frankas.num_dof
        self.franka_dof_pos = torch.zeros(
            (self.num_envs, self.num_franka_dofs), device=self._device
        )
        dof_limits = self.__frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(
            device=self._device
        )
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(
            device=self._device
        )
        self.franka_dof_speed_scales = torch.ones_like(
            self.franka_dof_lower_limits
        )
        self.franka_dof_speed_scales[self.__frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs),
            dtype=torch.float,
            device=self._device,
        )

        # caching defaults so that we can reset
        (
            self.default_box_pos,
            self.default_box_rot,
        ) = self.__boxes.get_world_poses()
        self.box_indices = torch.arange(
            self._num_envs, device=self._device
        ).view(self._num_envs)

        # reset all envs before we start
        reset_env_ids = torch.arange(
            self._num_envs, dtype=torch.int64, device=self._device
        )
        self.__reset_idx(reset_env_ids)

    # TODO: Read (ok)
    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        * INFO: Called from VecEnvBase before each simulation step,
        * and will pass in actions from the RL policy as an argument.
        * Convert actions to forces and apply on robot
        """
        if not self._env._world.is_playing():
            return

        # get all non-zero indices of reset buffer and request reset
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # TODO: Undo
            False and self.__reset_idx(reset_env_ids)

        self.__actions = actions.clone().to(self._device)
        targets = (
            self.franka_dof_targets
            + self.franka_dof_speed_scales
            * self.dt
            * self.__actions
            * self.action_scale
        )

        # clamp targets to joint limits
        self.franka_dof_targets[:] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )

        env_ids_int32 = torch.arange(
            self.__frankas.count, dtype=torch.int32, device=self._device
        )

        # apply actions to all envs
        self.__frankas.set_joint_position_targets(
            self.franka_dof_targets, indices=env_ids_int32
        )

    # TODO: Read and fix
    def get_observations(self) -> dict:
        """
        * INFO: Process physics and get observation for RL policy
        *
        * Needs to return information taken up by VecEnvBase and passed
        * to the RL policy
        """
        hand_pos, hand_rot = self.__frankas._hands.get_world_poses(clone=False)
        franka_dof_pos = self.__frankas.get_joint_positions(clone=False)
        franka_dof_vel = self.__frankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        (
            self.franka_grasp_rot,
            self.franka_grasp_pos,
        ) = self.__compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.__franka_local_grasp_rot,
            self.__franka_local_grasp_pos,
        )

        (
            self.franka_lfinger_pos,
            self.franka_lfinger_rot,
        ) = self.__frankas._lfingers.get_world_poses(clone=False)
        (
            self.franka_rfinger_pos,
            self.franka_rfinger_rot,
        ) = self.__frankas._lfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )

        # TODO: to_target will be target position - box position
        # to_target = self.drawer_grasp_pos - self.franka_grasp_pos
        to_target = 0

        # TODO: observation buffer needs to be modified
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                to_target,
            ),
            dim=-1,
        )

        observations = {self.__frankas.name: {"obs_buf": self.obs_buf}}
        return observations

    # TODO: Read and fix
    def calculate_metrics(self) -> None:
        """
        * INFO: Compute rewards for RL policy
        *
        * Needs to return information taken up by VecEnvBase and passed
        * to the RL policy
        """
        self.rew_buf[:] = self.__compute_franka_reward(
            self.__actions,
            self.action_penalty_scale,
        )

    def is_done(self) -> None:
        """
        * INFO: Check if episode is done or bad state is reached
        * Sets a boolean tensor of shape (num_envs,) to indicate
        * which environments are done and need a reset
        *
        * Needs to return information taken up by VecEnvBase and passed
        * to the RL policy
        """
        self.reset_buf = torch.where(
            self.progress_buf
            >= self._max_episode_length
            - 1,  # progress_buf incremented in post_physics
            # progress_buf is reset for each env in reset_idx
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    # ---------------------------------------------------------------------- #
    #                            HELPER FUNCTIONS                            #
    # ---------------------------------------------------------------------- #

    def __add_franka_to_stage(self):
        # * INFO: Adds franka to stage (uses a default USD)
        franka = Franka(
            prim_path=self.default_zero_env_path + "/franka", name="franka"
        )

        # TODO: What is going on here?
        self._sim_config.apply_articulation_settings(
            "franka",
            get_prim_at_path(franka.prim_path),
            self._sim_config.parse_actor_config("franka"),
        )

    def __add_box_to_stage(self) -> None:
        box_color = torch.tensor([0.2, 0.4, 0.6])
        box_pos = torch.tensor([0.3, 0.3, 0.0])

        box_size = 0.08

        box = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/box",
            name="box",
            color=box_color,
            size=box_size,
            density=100.0,
            translation=box_pos,
        )

        self._sim_config.apply_articulation_settings(
            "box",
            get_prim_at_path(box.prim_path),
            self._sim_config.parse_actor_config("box"),
        )

    # TODO: Read (ok), still need to under franka reset code
    def __reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = tensor_clamp(
            self.__franka_default_dof_pos.unsqueeze(0)
            + 0.25
            * (
                torch.rand(
                    (len(env_ids), self.num_franka_dofs), device=self._device
                )
                - 0.5
            ),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros(
            (num_indices, self.__frankas.num_dof), device=self._device
        )
        dof_vel = torch.zeros(
            (num_indices, self.__frankas.num_dof), device=self._device
        )
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos
        self.__frankas.set_joint_position_targets(
            self.franka_dof_targets[env_ids], indices=indices
        )
        self.__frankas.set_joint_positions(dof_pos, indices=indices)
        self.__frankas.set_joint_velocities(dof_vel, indices=indices)

        # reset boxes
        self.__boxes.set_world_poses(
            self.default_box_pos[self.box_indices[env_ids].flatten()],
            self.default_box_rot[self.box_indices[env_ids].flatten()],
            self.box_indices[env_ids].flatten().to(torch.int32),
        )

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    # TODO: Read: need to add box pos and rot calculation in this
    def __compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos

    # TODO: Read and remove drawer
    def __compute_franka_reward(
        self,
        actions: torch.Tensor,
        action_penalty_scale: float,
    ) -> torch.Tensor:
        # TODO: Add box pos and rot metric

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        rewards = -action_penalty_scale * action_penalty

        return rewards


def get_env_local_pose(env_pos, xformable, device):
    """Compute pose in env-local coordinates"""
    # env_pos here is only for one env, not all envs
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    world_pos = world_transform.ExtractTranslation()
    world_quat = world_transform.ExtractRotationQuat()

    px = world_pos[0] - env_pos[0]
    py = world_pos[1] - env_pos[1]
    pz = world_pos[2] - env_pos[2]
    qx = world_quat.imaginary[0]
    qy = world_quat.imaginary[1]
    qz = world_quat.imaginary[2]
    qw = world_quat.real

    return torch.tensor(
        [px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float
    )
