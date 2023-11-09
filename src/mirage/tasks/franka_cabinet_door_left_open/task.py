import torch
import numpy as np
from omni.isaac.core.utils.extensions import enable_extension
from ..franka_cabinet_drawer_top_open.task import (
    CustomTask as BaseTask,
)
from omni.isaac.core.utils.stage import get_current_stage
from .reward import compute_open_door_left_reward
from ...utils.transforms_utils import (
    compute_grasp_transforms,
    get_robot_local_grasp_transforms,
)
from .config import JOINT_INDEX
from .reward import LEFT_DOOR_JOINT_ALMOST_OPEN

enable_extension("omni.replicator.isaac")  # required by OIGE
enable_extension("omni.kit.window.viewport")  # required by OIGE


class CustomTask(BaseTask):
    def init_data(self) -> None:
        # * Used for computing grasp transforms in get_observations()
        (
            self.robot_local_grasp_pos,
            self.robot_local_grasp_rot,
        ) = get_robot_local_grasp_transforms(self, get_current_stage())

        # * Expected grasp pose franka to open the door
        door_local_grasp_pose = torch.tensor(
            [0.02, 0.35, 0.18, 1.0, 0.0, 0.0, 0.0], device=self._device
        )
        self.door_local_grasp_pos = door_local_grasp_pose[0:3].repeat(
            (self._num_envs, 1)
        )  # xyz
        self.door_local_grasp_rot = door_local_grasp_pose[3:7].repeat(
            (self._num_envs, 1)
        )  # wxyz

        # * gripper forward, up axis and door inward, up axis, used for
        # * aligning gripper with door
        self.gripper_forward_axis = torch.tensor(
            [0, 0, 1], device=self._device, dtype=torch.float
        ).repeat((self._num_envs, 1))
        self.door_inward_axis = torch.tensor(
            [-1, 0, 0], device=self._device, dtype=torch.float
        ).repeat((self._num_envs, 1))
        self.gripper_up_axis = torch.tensor(
            [0, 1, 0], device=self._device, dtype=torch.float
        ).repeat((self._num_envs, 1))
        self.door_up_axis = torch.tensor(
            [0, 0, 1], device=self._device, dtype=torch.float
        ).repeat((self._num_envs, 1))

        # * Articulation works in radians, we keep degrees for convenience
        self.robot_default_dof_pos = torch.tensor(
            np.radians(
                [0, -45, 0, -135, 0, 90, 45, 0, 0]
            ),  # TODO: gripper pos set to 0 0, should we change?
            device=self._device,
            dtype=torch.float32,
        )
        self.actions = torch.zeros(
            (self._num_envs, self.num_actions), device=self._device
        )

    def get_observations(self) -> dict:
        # -------------- COMPUTATION FOR OBSERVATION BUFFER ----------------#

        hand_pos, hand_rot = self._robots._hands.get_world_poses(clone=False)
        door_pos, door_rot = self._cabinets._left_door.get_world_poses(
            clone=False
        )

        robot_dof_pos = self._robots.get_joint_positions(clone=False)
        robot_dof_vel = self._robots.get_joint_velocities(clone=False)

        cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        cabinet_dof_vel = self._cabinets.get_joint_velocities(clone=False)

        (
            robot_grasp_rot,
            robot_grasp_pos,
            door_grasp_rot,
            door_grasp_pos,
        ) = compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot,
            self.robot_local_grasp_pos,
            door_rot,
            door_pos,
            self.door_local_grasp_rot,
            self.door_local_grasp_pos,
        )

        to_target = door_grasp_pos - robot_grasp_pos

        dof_pos_scaled = (
            2.0
            * (robot_dof_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        dof_vel_scaled = robot_dof_vel * self.dof_vel_scale

        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,  # size 9
                dof_vel_scaled,  # size 9
                to_target,  # size 3
                cabinet_dof_pos[:, JOINT_INDEX].unsqueeze(
                    -1
                ),  # top door joint pos - size 1
                cabinet_dof_vel[:, JOINT_INDEX].unsqueeze(
                    -1
                ),  # top door joint vel - size 1
            ),
            dim=-1,
        )

        # -------------- PRE-COMPUTATION FOR REWARD BUFFER ------------- #
        self.robot_dof_pos = robot_dof_pos
        self.cabinet_dof_pos = cabinet_dof_pos
        self.robot_grasp_pos, self.robot_grasp_rot = (
            robot_grasp_pos,
            robot_grasp_rot,
        )
        self.door_grasp_pos, self.door_grasp_rot = (
            door_grasp_pos,
            door_grasp_rot,
        )
        (
            self.robot_lfinger_pos,
            self.robot_lfinger_rot,
        ) = self._robots._lfingers.get_world_poses(clone=False)
        (
            self.robot_rfinger_pos,
            self.robot_rfinger_rot,
        ) = self._robots._lfingers.get_world_poses(clone=False)

        return {self._robots.name: {"obs_buf": self.obs_buf}}

    def is_done(self) -> None:
        # reset if door is open or max length reached
        self.reset_buf = torch.where(
            self.cabinet_dof_pos[:, 0] < LEFT_DOOR_JOINT_ALMOST_OPEN,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = compute_open_door_left_reward(
            self.actions,
            self.robot_dof_pos,
            self.cabinet_dof_pos,
            self.robot_grasp_pos,
            self.door_grasp_pos,
            self.robot_grasp_rot,
            self.door_grasp_rot,
            self.robot_lfinger_pos,
            self.robot_rfinger_pos,
            self.gripper_forward_axis,
            self.door_inward_axis,
            self.gripper_up_axis,
            self.door_up_axis,
            self._num_envs,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.around_handle_reward_scale,
            self.open_reward_scale,
            self.finger_dist_reward_scale,
            self.action_penalty_scale,
            self.finger_close_reward_scale,
        )
