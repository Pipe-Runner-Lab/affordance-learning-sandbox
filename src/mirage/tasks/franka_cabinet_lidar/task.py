from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.replicator.isaac")  # required by OIGE
enable_extension("omni.kit.window.viewport")  # required by OIGE

import torch  # noqa: E402
import numpy as np  # noqa: E402
from omniisaacgymenvs.tasks.base.rl_task import RLTask  # noqa: E402
from omni.isaac.core.utils.stage import get_current_stage  # noqa: E402
from ..franka_cabinet.asset_loader import spawn_cabinet  # noqa: E402
from .asset_loader import spawn_lidar_robot as spawn_robot  # noqa: E402
from ..franka_cabinet.utils import (  # noqa: E402
    compute_grasp_transforms,
    compute_reward,
    get_robot_local_grasp_transforms,
)


# provides inverse kinematics utils for cartesian space
# from skrl.utils import omniverse_isaacgym_utils

# post_physics_step calls
# - get_observations()
# - get_states()
# - calculate_metrics()
# - is_done()
# - get_extras()


class FrankaCabinetLidarTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self.dt = 1 / 60.0

        # observation and action space
        self._num_observations = 23
        self._num_actions = 9

        RLTask.__init__(self, name, env)

    # isaac extension workflow API
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.top_drawer_joint_threshold = self._task_cfg["env"][
            "topDrawerJointThreshold"
        ]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"][
            "aroundHandleRewardScale"
        ]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"][
            "fingerDistRewardScale"
        ]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"][
            "fingerCloseRewardScale"
        ]

    def set_up_scene(self, scene) -> None:
        self.get_robots = spawn_robot(self)
        self.get_cabinets = spawn_cabinet(self)

        super().set_up_scene(scene, filter_collisions=False)

        # articulation views views
        self._robots = self.get_robots(scene)
        self._cabinets = self.get_cabinets(scene)

        # lidar view
        from .view import LidarView  # noqa: E402

        self.lidars = {}
        for i in range(self._num_envs):
            self.lidars[i] = LidarView(
                prim_paths=f"/World/envs/env_{i}/franka/panda_hand/Lidar",
                name=f"lidar_view_{i}",
            )
            scene.add(self.lidars[i])

        self.init_data()

    # isaac extension workflow API
    def initialize_views(self, scene):
        super().initialize_views(scene)

        self._robots = self.get_robots(scene)
        self._cabinets = self.get_cabinets(scene)

        self.init_data()

    def init_data(self) -> None:
        # * Used for computing grasp transforms in get_observations()
        (
            self.robot_local_grasp_pos,
            self.robot_local_grasp_rot,
        ) = get_robot_local_grasp_transforms(self, get_current_stage())

        # * Expected grasp pose franka to open the drawer
        drawer_local_grasp_pose = torch.tensor(
            [0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device
        )
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat(
            (self._num_envs, 1)
        )
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat(
            (self._num_envs, 1)
        )

        # * gripper forward, up axis and drawer inward, up axis, used for
        # * aligning gripper with drawer
        self.gripper_forward_axis = torch.tensor(
            [0, 0, 1], device=self._device, dtype=torch.float
        ).repeat((self._num_envs, 1))
        self.drawer_inward_axis = torch.tensor(
            [-1, 0, 0], device=self._device, dtype=torch.float
        ).repeat((self._num_envs, 1))
        self.gripper_up_axis = torch.tensor(
            [0, 1, 0], device=self._device, dtype=torch.float
        ).repeat((self._num_envs, 1))
        self.drawer_up_axis = torch.tensor(
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

    def post_reset(self):
        self.num_robot_dofs = self._robots.num_dof
        self.robot_dof_pos = torch.zeros(
            (self.num_envs, self.num_robot_dofs), device=self._device
        )
        dof_limits = self._robots.get_dof_limits()

        # used for clamping dof values when forcing them
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(
            device=self._device
        )
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(
            device=self._device
        )

        # Setting velocity scales for each dof
        # (arm scale is 1, gripper scale is 0.1)
        # * torch.ones_like returns a tensor with the same
        # * size as the input tensor filled with 1
        self.robot_dof_speed_scales = torch.ones_like(
            self.robot_dof_lower_limits
        )
        self.robot_dof_speed_scales[self._robots.gripper_indices] = 0.1

        self.robot_dof_targets = torch.zeros(
            (self._num_envs, self.num_robot_dofs),
            dtype=torch.float,
            device=self._device,
        )

        # reset all envs
        reset_env_ids = torch.arange(
            self._num_envs, dtype=torch.int64, device=self._device
        )
        self.reset_idx(reset_env_ids)

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # mandatory call for lidar to work
        self._env.render()

        # ! WHY?
        if not self._env._world.is_playing():
            return

        # reset envs
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        env_ids_int32 = torch.arange(
            self._robots.count, dtype=torch.int32, device=self._device
        )

        # update dof targets based on actions, dt and dof speed scales
        # TODO: simply using +=
        temp_dof_targets = (
            self.robot_dof_targets
            + self.robot_dof_speed_scales
            * self.dt
            * self.actions
            * self.action_scale
        )

        # clamp dof targets to limits
        # TODO: Why [:]
        self.robot_dof_targets[:] = torch.clamp(
            temp_dof_targets,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

        # apply joint targets to robots in all envs
        self._robots.set_joint_position_targets(
            self.robot_dof_targets, indices=env_ids_int32
        )

    def get_observations(self) -> dict:
        print(self.lidars[0].get_current_frame())

        # -------------- COMPUTATION FOR OBSERVATION BUFFER ----------------#

        hand_pos, hand_rot = self._robots._hands.get_world_poses(clone=False)
        drawer_pos, drawer_rot = self._cabinets._drawers.get_world_poses(
            clone=False
        )

        robot_dof_pos = self._robots.get_joint_positions(clone=False)
        robot_dof_vel = self._robots.get_joint_velocities(clone=False)

        (
            franka_lfinger_pos,
            franka_lfinger_rot,
        ) = self._robots._lfingers.get_world_poses(clone=False)

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
                cabinet_dof_pos[:, 3].unsqueeze(
                    -1
                ),  # drawer joint pos - size 1
                cabinet_dof_vel[:, 3].unsqueeze(
                    -1
                ),  # drawer joint vel - size 1
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

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = compute_reward(
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

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(
            self.cabinet_dof_pos[:, 3] > self.top_drawer_joint_threshold,
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
        self._cabinets.set_joint_positions(
            torch.zeros_like(
                self._cabinets.get_joint_positions(clone=False)[env_ids]
            ),
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
