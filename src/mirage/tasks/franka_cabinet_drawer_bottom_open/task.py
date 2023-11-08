import torch
from omni.isaac.core.utils.extensions import enable_extension
from ..franka_cabinet_drawer_top_open.task import (
    CustomTask as BaseTask,
)
from ...utils.transforms_utils import compute_grasp_transforms

enable_extension("omni.replicator.isaac")  # required by OIGE
enable_extension("omni.kit.window.viewport")  # required by OIGE


class CustomTask(BaseTask):
    def get_observations(self) -> dict:
        # -------------- COMPUTATION FOR OBSERVATION BUFFER ----------------#

        hand_pos, hand_rot = self._robots._hands.get_world_poses(clone=False)
        (
            drawer_pos,
            drawer_rot,
        ) = self._cabinets._bottom_drawers.get_world_poses(clone=False)

        robot_dof_pos = self._robots.get_joint_positions(clone=False)
        robot_dof_vel = self._robots.get_joint_velocities(clone=False)

        cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        cabinet_dof_vel = self._cabinets.get_joint_velocities(clone=False)

        (
            robot_grasp_rot,
            robot_grasp_pos,
            drawer_grasp_rot,
            drawer_grasp_pos,
        ) = compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot,
            self.robot_local_grasp_pos,
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot,
            self.drawer_local_grasp_pos,
        )

        to_target = drawer_grasp_pos - robot_grasp_pos

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
                cabinet_dof_pos[:, 2].unsqueeze(
                    -1
                ),  # bottom drawer joint pos - size 1
                cabinet_dof_vel[:, 2].unsqueeze(
                    -1
                ),  # bottom drawer joint vel - size 1
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
        self.drawer_grasp_pos, self.drawer_grasp_rot = (
            drawer_grasp_pos,
            drawer_grasp_rot,
        )
        hand_pos, hand_rot = self._robots._hands.get_world_poses(clone=False)
        (
            self.robot_lfinger_pos,
            self.robot_lfinger_rot,
        ) = self._robots._lfingers.get_world_poses(clone=False)
        (
            self.robot_rfinger_pos,
            self.robot_rfinger_rot,
        ) = self._robots._lfingers.get_world_poses(clone=False)

        return {self._robots.name: {"obs_buf": self.obs_buf}}
