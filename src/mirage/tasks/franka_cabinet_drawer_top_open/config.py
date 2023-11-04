TASK_CFG = {
    "test": False,
    "device_id": 0,
    "headless": True,  # get_env_instance(headless=False) overrides this
    "sim_device": "gpu",
    "enable_livestream": False,
    "warp": False,
    "seed": 0,
    "task": {
        "name": "AffordanceBlockPickPlace",
        "physics_engine": "physx",
        "env": {
            "numEnvs": 1024,
            "envSpacing": 3,
            "episodeLength": 500,
            "enableDebugVis": False,  # VS Code debugger
            "clipObservations": 5.0,
            "clipActions": 1.0,
            "controlFrequencyInv": 2,
            "controlSpace": "joint",  # "joint" or "cartesian"
            # scales
            "actionScale": 7.5,
            "dofVelocityScale": 0.1,
            "distRewardScale": 2.0,
            "rotRewardScale": 0.5,
            "aroundHandleRewardScale": 10.0,
            "openRewardScale": 7.5,
            "fingerDistRewardScale": 100.0,
            "actionPenaltyScale": 0.01,
            "fingerCloseRewardScale": 10.0,
            "topDrawerJointThreshold": 0.39,
        },
        "sim": {
            # simulation dt is different from control dt
            "dt": 0.0083,  # 1 / 120
            "use_gpu_pipeline": True,
            "gravity": [0.0, 0.0, -9.81],
            "add_ground_plane": True,
            "use_flatcache": True,
            "enable_scene_query_support": False,
            "enable_cameras": False,
            "add_distant_light": False,
            "use_fabric": True,
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
                "solver_position_iteration_count": 12,
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
                "fixed_base": True,  # ! changed to true
                "enable_self_collisions": False,
                "enable_gyroscopic_forces": True,
                "solver_position_iteration_count": 12,
                "solver_velocity_iteration_count": 1,
                "sleep_threshold": 0.005,
                "stabilization_threshold": 0.001,
                "density": -1,
                "max_depenetration_velocity": 1000.0,
                "contact_offset": 0.005,  # ! not in original
                "rest_offset": 0.0,  # ! not in original
            },
            "cabinet": {
                # -1 to use default values
                "override_usd_defaults": False,
                "enable_self_collisions": False,
                "enable_gyroscopic_forces": True,
                # also in stage params
                # per-actor
                "solver_position_iteration_count": 12,
                "solver_velocity_iteration_count": 1,
                "sleep_threshold": 0.0,
                "stabilization_threshold": 0.001,
                # per-body
                "density": -1,
                "max_depenetration_velocity": 1000.0,
            },
        },
    },
}
