from pxr import Usd
from pxr import UsdGeom
from omniisaacgymenvs.tasks.base.rl_task import RLTask
import torch
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse


def get_robot_local_grasp_transforms(
    task: RLTask, stage: Usd.Stage  # pylint: disable=no-member
):
    env_local_hand_pose = get_env_local_pose(
        task._env_pos[0],
        UsdGeom.Xformable(  # pylint: disable=no-member
            stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")
        ),
        task._device,
    )
    env_local_lfinger_pose = get_env_local_pose(
        task._env_pos[0],
        UsdGeom.Xformable(  # pylint: disable=no-member
            stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")
        ),
        task._device,
    )
    env_local_rfinger_pose = get_env_local_pose(
        task._env_pos[0],
        UsdGeom.Xformable(  # pylint: disable=no-member
            stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")
        ),
        task._device,
    )

    env_local_finger_pose = torch.zeros(7, device=task._device)
    env_local_finger_pose[0:3] = (
        env_local_lfinger_pose[0:3] + env_local_rfinger_pose[0:3]
    ) / 2.0  # xyz position
    env_local_finger_pose[3:7] = env_local_lfinger_pose[3:7]  # wxyz quaternion

    # finding inverse translation and rotation of hand
    env_local_hand_pose_inv_rot, env_local_hand_pose_inv_pos = tf_inverse(
        env_local_hand_pose[3:7], env_local_hand_pose[0:3]
    )

    # This applies inverse, thus we get the finger pose in the hand frame
    robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
        env_local_hand_pose_inv_rot,
        env_local_hand_pose_inv_pos,
        env_local_finger_pose[3:7],
        env_local_finger_pose[0:3],
    )
    robot_local_pose_pos += torch.tensor(
        [0, 0.04, 0], device=task._device
    )  # TODO: Why?

    # repeat for all envs
    robot_local_grasp_pos = robot_local_pose_pos.repeat((task._num_envs, 1))
    robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat(
        (task._num_envs, 1)
    )

    return robot_local_grasp_pos, robot_local_grasp_rot


def get_env_local_pose(env_pos, xformable, device):
    """Compute pose in env-local coordinates by subtracting env_pos from world_pos"""
    # env_pos here is only for one env, not all envs
    # https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/transforms/get-world-transforms.html#get-the-world-space-transforms-for-a-prim
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
    robot_local_grasp_rot,
    robot_local_grasp_pos,
    drawer_rot,
    drawer_pos,
    drawer_local_grasp_rot,
    drawer_local_grasp_pos,
):
    global_robot_rot, global_robot_pos = tf_combine(
        hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos
    )
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
    )

    return (
        global_robot_rot,
        global_robot_pos,
        global_drawer_rot,
        global_drawer_pos,
    )
