from pxr import Usd
from pxr import UsdGeom
from omniisaacgymenvs.tasks.base.rl_task import RLTask
import torch
from omni.isaac.core.utils.torch.transformations import (
    tf_combine,
    tf_inverse,
    tf_vector,
)


def get_robot_local_grasp_transforms(
    task: RLTask, stage: Usd.Stage  # pylint: disable=no-member
):
    hand_pose = get_env_local_pose(
        task._env_pos[0],
        UsdGeom.Xformable(  # pylint: disable=no-member
            stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")
        ),
        task._device,
    )
    lfinger_pose = get_env_local_pose(
        task._env_pos[0],
        UsdGeom.Xformable(  # pylint: disable=no-member
            stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")
        ),
        task._device,
    )
    rfinger_pose = get_env_local_pose(
        task._env_pos[0],
        UsdGeom.Xformable(  # pylint: disable=no-member
            stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")
        ),
        task._device,
    )

    finger_pose = torch.zeros(7, device=task._device)
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
    franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=task._device)

    # repeat for all envs
    franka_local_grasp_pos = franka_local_pose_pos.repeat((task._num_envs, 1))
    franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat(
        (task._num_envs, 1)
    )

    return franka_local_grasp_pos, franka_local_grasp_rot


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


# TODO: What does it do?
def compute_grasp_transforms(
    hand_rot,
    hand_pos,
    franka_local_grasp_rot,
    franka_local_grasp_pos,
    drawer_rot,
    drawer_pos,
    drawer_local_grasp_rot,
    drawer_local_grasp_pos,
):
    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
    )
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
    )

    return (
        global_franka_rot,
        global_franka_pos,
        global_drawer_rot,
        global_drawer_pos,
    )


def compute_reward(
    # entities
    actions,
    robot_dof_pos,
    cabinet_dof_pos,
    robot_grasp_pos,
    drawer_grasp_pos,
    robot_grasp_rot,
    drawer_grasp_rot,
    robot_lfinger_pos,
    robot_rfinger_pos,
    # constants
    gripper_forward_axis,
    drawer_inward_axis,
    gripper_up_axis,
    drawer_up_axis,
    num_envs,
    dist_reward_scale,
    rot_reward_scale,
    around_handle_reward_scale,
    open_reward_scale,
    finger_dist_reward_scale,
    action_penalty_scale,
    finger_close_reward_scale,
):
    # distance from hand to the drawer
    d = torch.norm(robot_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d**2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(robot_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
    axis3 = tf_vector(robot_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

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
    # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    rot_reward = 0.5 * (
        torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2
    )

    # bonus if left finger is above the drawer handle and right below
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(
        robot_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
        torch.where(
            robot_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
            around_handle_reward + 0.5,
            around_handle_reward,
        ),
        around_handle_reward,
    )
    # reward for distance of each finger from the drawer
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(robot_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    rfinger_dist = torch.abs(robot_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    finger_dist_reward = torch.where(
        robot_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
        torch.where(
            robot_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
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
    open_reward = (
        cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]
    )  # drawer_top_joint

    rewards = (
        dist_reward_scale * dist_reward
        + rot_reward_scale * rot_reward
        + around_handle_reward_scale * around_handle_reward
        + open_reward_scale * open_reward
        + finger_dist_reward_scale * finger_dist_reward
        - action_penalty_scale * action_penalty
        + finger_close_reward * finger_close_reward_scale
    )

    # bonus for opening drawer properly
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
    rewards = torch.where(
        cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards
    )
    rewards = torch.where(
        cabinet_dof_pos[:, 3] > 0.39,
        rewards + (2.0 * around_handle_reward),
        rewards,
    )

    # # prevent bad style in opening drawer
    # rewards = torch.where(franka_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
    #                       torch.ones_like(rewards) * -1, rewards)
    # rewards = torch.where(franka_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
    #                       torch.ones_like(rewards) * -1, rewards)

    return rewards
