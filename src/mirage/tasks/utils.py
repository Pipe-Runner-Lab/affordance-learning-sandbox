from pxr import Usd
from pxr import UsdGeom
from omniisaacgymenvs.tasks.base.rl_task import RLTask
import torch
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse


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
