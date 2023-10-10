from .asset_loader import spawn_robot, spawn_target
from .utils import get_robot_local_grasp_transforms
import torch
import numpy as np
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.replicator.isaac")  # required by OIGE

from omniisaacgymenvs.tasks.base.rl_task import RLTask  # noqa: E402


# provides inverse kinematics utils for cartesian space
# from skrl.utils import omniverse_isaacgym_utils

# post_physics_step calls
# - get_observations()
# - get_states()
# - calculate_metrics()
# - is_done()
# - get_extras()


TASK_CFG = {
    "test": False,
    "device_id": 0,
    "headless": True,
    "sim_device": "gpu",
    "enable_livestream": False,
    "task": {
        "name": "AffordanceBlockPickPlace",
        "physics_engine": "physx",
        "env": {
            "numEnvs": 2,
            "envSpacing": 3,
            "episodeLength": 500,  # ! different
            "enableDebugVis": False,  # VS Code debugger
            "clipObservations": 5.0,  # ! different
            "clipActions": 1.0,
            "controlFrequencyInv": 2,  # ! different : 60 Hz
            "actionScale": 7.5,
            "dofVelocityScale": 0.1,
            # ! params missing
        },
        "sim": {
            "dt": 0.0083,  # 1 / 120
            "use_gpu_pipeline": True,
            "gravity": [0.0, 0.0, -9.81],
            "add_ground_plane": True,
            "use_flatcache": True,
            "enable_scene_query_support": False,
            "enable_cameras": False,
            "default_physics_material": {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0,
            },
            "physx": {
                # * Number of worker threads per scene used by
                # * PhysX - for CPU PhysX only
                "worker_thread_count": 4,
                # * 0: pgs, 1 : Temporal Gauss-Seidel (TGS) solver
                "solver_type": 1,
                "use_gpu": True,
                "solver_position_iteration_count": 4,  # ! different
                "solver_velocity_iteration_count": 1,
                "contact_offset": 0.005,
                "rest_offset": 0.0,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.04,
                "friction_correlation_distance": 0.025,
                "enable_sleeping": True,
                "enable_stabilization": True,
                "max_depenetration_velocity": 1000.0,
                # GPU buffers
                "gpu_max_rigid_contact_count": 524288,
                "gpu_max_rigid_patch_count": 33554432,
                "gpu_found_lost_pairs_capacity": 524288,
                "gpu_found_lost_aggregate_pairs_capacity": 262144,
                "gpu_total_aggregate_pairs_capacity": 1048576,
                "gpu_max_soft_body_contacts": 1048576,
                "gpu_max_particle_contacts": 1048576,
                "gpu_heap_capacity": 33554432,
                "gpu_temp_buffer_capacity": 16777216,
                "gpu_max_num_partitions": 8,
            },
            "robot": {
                "override_usd_defaults": False,
                "fixed_base": False,  # ! Why is this false?
                "enable_self_collisions": False,
                "enable_gyroscopic_forces": True,
                "solver_position_iteration_count": 4,  # ! different
                "solver_velocity_iteration_count": 1,
                "sleep_threshold": 0.005,
                "stabilization_threshold": 0.001,
                "density": -1,
                "max_depenetration_velocity": 1000.0,
                "contact_offset": 0.005,  # ! not in original
                "rest_offset": 0.0,  # ! not in original
            },
            "target": {
                "override_usd_defaults": False,
                "fixed_base": True,  # ! Why is this true?
                "enable_self_collisions": False,
                "enable_gyroscopic_forces": True,
                "solver_position_iteration_count": 4,  # ! different
                "solver_velocity_iteration_count": 1,
                "sleep_threshold": 0.005,
                "stabilization_threshold": 0.001,
                "density": -1,
                "max_depenetration_velocity": 1000.0,
                "contact_offset": 0.005,
                "rest_offset": 0.0,
            },
        },
    },
}


class ReachingFrankaTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.dt = 1 / 120.0

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._action_scale = self._task_cfg["env"]["actionScale"]
        self._dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        # observation and action space
        self._num_observations = 18  # TODO: Change based on buffer size
        self._num_actions = 9  # TODO: Increase from 7 to 9 for gripper

        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        get_robots = spawn_robot(self)
        get_targets = spawn_target(self)

        super().set_up_scene(scene)

        # articulation views views
        self._robots = get_robots(scene)
        self._targets = get_targets(scene)

        self.init_data()

    def init_data(self) -> None:
        # * Used for computing grasp transforms in get_observations()
        (
            self.robot_local_grasp_pos,
            self.robot_local_grasp_rot,
        ) = get_robot_local_grasp_transforms(self, get_current_stage())

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

        # Caching default box poses
        (
            self.default_target_pos,
            self.default_target_rot,
        ) = self._targets.get_world_poses()

        # reset all envs
        reset_env_ids = torch.arange(
            self._num_envs, dtype=torch.int64, device=self._device
        )
        self.reset_idx(reset_env_ids)

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # reset envs
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        env_ids_int32 = torch.arange(
            self._robots.count, dtype=torch.int32, device=self._device
        )

        targets = (
            self.robot_dof_targets
            + self.robot_dof_speed_scales
            * self.dt
            * self.actions
            * self._action_scale
        )

        self.robot_dof_targets = torch.clamp(
            targets,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

        # apply joint targets to robots in all envs
        self._robots.set_joint_position_targets(
            self.robot_dof_targets, indices=env_ids_int32
        )

    def get_observations(self) -> dict:
        # -------------------- COMPUTATION FOR OBSERVATION BUFFER -------------------- #

        robot_dof_pos = self._robots.get_joint_positions(clone=False)
        robot_dof_vel = self._robots.get_joint_velocities(clone=False)
        target_pos, target_rot = self._targets.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (robot_dof_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        dof_vel_scaled = robot_dof_vel * self._dof_vel_scale

        # assuming sensor noise
        generalization_noise = (
            torch.rand((dof_vel_scaled.shape[0], 7), device=self._device) + 0.5
        )

        # Look at observation table:
        # https://skrl.readthedocs.io/en/latest/intro/examples.html#real-world-examples
        # TODO: Use torch.cat
        # TODO: Add gripper pos and vel
        # TODO: Target rot should also be added
        self.obs_buf[:, 0] = self.progress_buf / self._max_episode_length
        self.obs_buf[:, 1:8] = dof_pos_scaled[:, :7]
        self.obs_buf[:, 8:15] = dof_vel_scaled[:, :7] * generalization_noise
        self.obs_buf[:, 15:18] = target_pos - self._env_pos

        # --------------------- PRE-COMPUTATION FOR REWARD BUFFER -------------------- #
        # TODO: Not being used in observation, only used for reward
        hand_pos, hand_rot = self._robots._hands.get_world_poses(clone=False)
        lfinger_pos, lfinger_rot = self._robots._lfingers.get_world_poses(
            clone=False
        )
        rfinger_pos, rfinger_rot = self._robots._rfingers.get_world_poses(
            clone=False
        )

        # compute distance for calculate_metrics() and is_done()
        # TODO: Change 0 to some target position
        self._computed_distance = torch.norm(0 - target_pos, dim=-1)

        return {self._robots.name: {"obs_buf": self.obs_buf}}

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = -self._computed_distance

    def is_done(self) -> None:
        # fill with 0
        self.reset_buf.fill_(0)

        # TODO: This needs to be adjusted based on task
        # target reached
        self.reset_buf = torch.where(
            self._computed_distance <= 0.035,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
        # max episode length
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

        # reset target with random noise added to position
        rand_pos = (
            torch.rand((len(env_ids), 3), device=self._device) - 0.5
        ) * 2 * torch.tensor(
            [0.25, 0.25, 0.10], device=self._device
        ) + torch.tensor(
            [0.50, 0.00, 0.20], device=self._device
        )

        # TODO: look into adding randomness
        self._targets.set_world_poses(
            # positions=rand_pos + self.default_target_pos[env_ids],
            positions=self.default_target_pos[env_ids],
            orientations=self.default_target_rot[env_ids],
            indices=indices,
        )

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
