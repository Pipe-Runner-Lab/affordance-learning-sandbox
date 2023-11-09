import torch
from omni.isaac.core.utils.extensions import enable_extension
from ..franka_cabinet_drawer_top_open.task import (
    CustomTask as BaseTask,
)
from .reward import compute_close_drawer_reward
from .config import JOINT_INDEX
from .reward import TOP_DRAWER_CLOSED, TOP_DRAWER_JOINT_ALMOST_OPEN

enable_extension("omni.replicator.isaac")  # required by OIGE
enable_extension("omni.kit.window.viewport")  # required by OIGE


class CustomTask(BaseTask):
    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(
            self.cabinet_dof_pos[:, JOINT_INDEX] < TOP_DRAWER_CLOSED,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def reset_idx(self, env_ids) -> None:
        indices = env_ids.to(dtype=torch.int32)

        # reset robot
        pos = torch.clamp(
            self.robot_default_dof_pos.unsqueeze(0)
            + 0.25
            * (
                torch.rand(
                    (len(env_ids), self.num_robot_dofs), device=self._device
                )
                - 0.5
            ),
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )
        dof_pos = torch.zeros(
            (len(indices), self._robots.num_dof), device=self._device
        )
        dof_pos[:, :] = pos
        dof_vel = torch.zeros(
            (len(indices), self._robots.num_dof), device=self._device
        )
        self.robot_dof_targets[env_ids, :] = pos
        self.robot_dof_pos[env_ids, :] = pos

        self._robots.set_joint_position_targets(
            self.robot_dof_targets[env_ids], indices=indices
        )
        self._robots.set_joint_positions(dof_pos, indices=indices)
        self._robots.set_joint_velocities(dof_vel, indices=indices)

        # reset cabinet
        default_cabinet_dof_pos = torch.zeros_like(
            self._cabinets.get_joint_positions(clone=False)[env_ids]
        )
        default_cabinet_dof_pos[:, JOINT_INDEX] = TOP_DRAWER_JOINT_ALMOST_OPEN
        self._cabinets.set_joint_positions(
            default_cabinet_dof_pos,
            indices=indices,
        )

        self._cabinets.set_joint_velocities(
            torch.zeros_like(
                self._cabinets.get_joint_velocities(clone=False)[env_ids]
            ),
            indices=indices,
        )

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = compute_close_drawer_reward(
            self.actions,
            self.robot_dof_pos,
            self.cabinet_dof_pos,
            self.robot_grasp_pos,
            self.drawer_grasp_pos,
            self.robot_grasp_rot,
            self.drawer_grasp_rot,
            self.robot_lfinger_pos,
            self.robot_rfinger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self._num_envs,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.around_handle_reward_scale,
            self.open_reward_scale,
            self.finger_dist_reward_scale,
            self.action_penalty_scale,
            self.finger_close_reward_scale,
        )
