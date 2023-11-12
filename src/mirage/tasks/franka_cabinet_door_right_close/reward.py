import torch
from omni.isaac.core.utils.torch.transformations import (
    tf_vector,
)
from .config import JOINT_INDEX

RIGHT_DOOR_JOINT_ALMOST_OPEN = 1.05  # almost open
RIGHT_DOOR_JOINT_PARTIALLY_OPEN = 0.6  # partially open
RIGHT_DOOR_ALMOST_CLOSED = 0.05  # almost closed
RIGHT_DOOR_CLOSED = 0.01  # closed


def compute_close_door_right_reward(
    # entities
    actions,
    robot_dof_pos,
    cabinet_dof_pos,
    robot_grasp_pos,
    door_grasp_pos,
    robot_grasp_rot,
    door_grasp_rot,
    robot_lfinger_pos,
    robot_rfinger_pos,
    # constants
    gripper_forward_axis,
    door_inward_axis,
    gripper_up_axis,
    door_up_axis,
    num_envs,
    dist_reward_scale,
    rot_reward_scale,
    around_handle_reward_scale,
    open_reward_scale,
    finger_dist_reward_scale,
    action_penalty_scale,
    finger_close_reward_scale,
):
    # distance from hand to the door
    d = torch.norm(robot_grasp_pos - door_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d**2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(robot_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(door_grasp_rot, door_inward_axis)
    axis3 = tf_vector(robot_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(door_grasp_rot, door_up_axis)

    dot1 = (
        torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1))
        .squeeze(-1)
        .squeeze(-1)
    )  # alignment of forward axis for gripper
    dot2 = (
        torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1))
        .squeeze(-1)
        .squeeze(-1)
    )  # alignment of up axis for gripper
    # reward for matching the orientation of the hand to the door
    # (fingers wrapped)
    rot_reward = 0.5 * (
        torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2
    )

    # bonus if left finger is above the door handle and right below
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(
        robot_lfinger_pos[:, 2] > door_grasp_pos[:, 2],
        torch.where(
            robot_rfinger_pos[:, 2] < door_grasp_pos[:, 2],
            around_handle_reward + 0.5,
            around_handle_reward,
        ),
        around_handle_reward,
    )
    # reward for distance of each finger from the door
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(robot_lfinger_pos[:, 2] - door_grasp_pos[:, 2])
    rfinger_dist = torch.abs(robot_rfinger_pos[:, 2] - door_grasp_pos[:, 2])
    finger_dist_reward = torch.where(
        robot_lfinger_pos[:, 2] > door_grasp_pos[:, 2],
        torch.where(
            robot_rfinger_pos[:, 2] < door_grasp_pos[:, 2],
            (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
            finger_dist_reward,
        ),
        finger_dist_reward,
    )

    finger_close_reward = torch.zeros_like(rot_reward)
    finger_close_reward = torch.where(
        d <= 0.03,
        (0.04 - robot_dof_pos[:, 7]) + (0.04 - robot_dof_pos[:, 8]),
        finger_close_reward,
    )

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions**2, dim=-1)

    # how far the cabinet has been opened out
    close_reward = (
        RIGHT_DOOR_JOINT_ALMOST_OPEN - cabinet_dof_pos[:, JOINT_INDEX]
    ) * around_handle_reward + (
        RIGHT_DOOR_JOINT_ALMOST_OPEN - cabinet_dof_pos[:, JOINT_INDEX]
    )  # door_left_joint

    rewards = (
        dist_reward_scale * dist_reward
        + rot_reward_scale * rot_reward
        + around_handle_reward_scale * around_handle_reward
        + open_reward_scale * close_reward
        + finger_dist_reward_scale * finger_dist_reward
        - action_penalty_scale * action_penalty
        + finger_close_reward * finger_close_reward_scale
    )

    # bonus for closing door properly
    rewards = torch.where(
        cabinet_dof_pos[:, JOINT_INDEX] < RIGHT_DOOR_JOINT_ALMOST_OPEN,
        rewards + 0.5,
        rewards,
    )
    rewards = torch.where(
        cabinet_dof_pos[:, JOINT_INDEX] < RIGHT_DOOR_JOINT_PARTIALLY_OPEN,
        rewards + around_handle_reward,
        rewards,
    )
    rewards = torch.where(
        cabinet_dof_pos[:, JOINT_INDEX] < RIGHT_DOOR_ALMOST_CLOSED,
        rewards + (2.0 * around_handle_reward),
        rewards,
    )

    return rewards
