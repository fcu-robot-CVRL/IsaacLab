# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # imu_scanner = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    # )
    imu_scanner_pelvis = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base"
    )
    imu_scanner_L_elbow = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base"
    )
    imu_scanner_R_elbow = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base"
    )
    imu_scanner_L_knee = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base"
    )
    imu_scanner_R_knee = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base"
    )
    # contact sensor
    contact_sensor_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base"
    )
    contact_sensor_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base"
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/g1/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    # 軀幹關節 - 最保守的控制
    # torso_joints = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=["torso_joint"], 
    #     scale=0.05  # 軀幹需要最穩定
    # )
    
    # # 髖關節 - 腿部核心，需要穩定但有一定靈活性
    # hip_joints = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=[
    #         "left_hip_pitch_joint", "right_hip_pitch_joint",
    #         "left_hip_roll_joint", "right_hip_roll_joint", 
    #         "left_hip_yaw_joint", "right_hip_yaw_joint"
    #     ], 
    #     scale=0.08
    # )
    
    # # 膝關節 - 步行的關鍵關節
    # knee_joints = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=["left_knee_joint", "right_knee_joint"], 
    #     scale=0.1
    # )
    
    # # 踝關節 - 平衡和地面接觸
    # ankle_joints = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=[
    #         "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    #         "left_ankle_roll_joint", "right_ankle_roll_joint"
    #     ], 
    #     scale=0.06
    # )
    
    # # 肩膀關節 - 平衡輔助
    # shoulder_joints = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=[
    #         "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    #         "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    #         "left_shoulder_yaw_joint", "right_shoulder_yaw_joint"
    #     ], 
    #     scale=0.04
    # )
    
    # # 手肘關節 - 較小的動作範圍
    # elbow_joints = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=[
    #         "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    #         "left_elbow_roll_joint", "right_elbow_roll_joint"
    #     ], 
    #     scale=0.03
    # )
    
    # # 手指關節 - 最小的動作範圍
    # finger_joints = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=[
    #         "left_zero_joint", "left_one_joint", "left_two_joint",
    #         "left_three_joint", "left_four_joint", "left_five_joint", "left_six_joint",
    #         "right_zero_joint", "right_one_joint", "right_two_joint", 
    #         "right_three_joint", "right_four_joint", "right_five_joint", "right_six_joint"
    #     ], 
    #     scale=0.01
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # imu_scanner -------------------------------------------------------
        # imuser_ang = ObsTerm(
        #     func=mdp.imusener_ang_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # imuser_lin = ObsTerm(
        #     func=mdp.imusener_lin_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # imu_scanner_pelvis_ang = ObsTerm(
        #     func=mdp.imusener_ang_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner_pelvis")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # imu_scanner_pelvis_lin = ObsTerm(
        #     func=mdp.imusener_lin_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner_pelvis")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # # imu_scanner_L_elbow -------------------------------------------------------
        # imu_scanner_L_elbow_ang = ObsTerm(
        #     func=mdp.imusener_ang_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner_L_elbow")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # imu_scanner_L_elbow_lin = ObsTerm(
        #     func=mdp.imusener_lin_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner_L_elbow")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # # imu_scanner_R_elbow -------------------------------------------------------
        # imu_scanner_R_elbow_ang = ObsTerm(
        #     func=mdp.imusener_ang_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner_R_elbow")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # imu_scanner_R_elbow_lin = ObsTerm(
        #     func=mdp.imusener_lin_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner_R_elbow")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # # imu_scanner_L_knee -------------------------------------------------------
        # imu_scanner_L_knee_ang = ObsTerm(
        #     func=mdp.imusener_ang_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner_L_knee")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # imu_scanner_L_knee_lin = ObsTerm(
        #     func=mdp.imusener_lin_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner_L_knee")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # # imu_scanner_R_knee -------------------------------------------------------
        # imu_scanner_R_knee_ang = ObsTerm(
        #     func=mdp.imusener_ang_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner_R_knee")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # imu_scanner_R_knee_lin = ObsTerm(
        #     func=mdp.imusener_lin_vel,
        #     params={"sensor_cfg": SceneEntityCfg("imu_scanner_R_knee")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # contact_sensor left_ankle -------------------------------------------------------
        # contact_sensor_left_ankle_roll_link = ObsTerm(
        #     func=mdp.contact_sensor_L,
        #     params={"sensor_cfg": SceneEntityCfg("contact_sensor_left")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )       
        
        # # contact_sensor right_ankle -------------------------------------------------------
        # contact_sensor_right_ankle_roll_link = ObsTerm(
        #     func=mdp.contact_sensor_R,
        #     params={"sensor_cfg": SceneEntityCfg("contact_sensor_right")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # 測試-------------------------------------------------------------------------
    # move_towards_target = RewTerm(
    #     func=mdp.move_towards_target, 
    #     weight=10.0, 
    #     params={"target_pos": (10.0, 0.0, 0.0)}  # 設定目標位置為 (10, 0, 0)
    # )

    # 新增的地板接觸獎勵
    # contact_ground_reward = RewTerm(
    #     func=mdp.contact_ground_reward, 
    #     weight=50.0, 
    #     params={
    #         "threshold": 1.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    #     }
    # )
    # 左腳接觸獎勵
    # contact_ground_left_foot = RewTerm(
    #     func=mdp.contact_ground_reward, 
    #     weight=50.0, 
    #     params={
    #         "threshold": 1.0,
    #         "sensor_cfg": SceneEntityCfg("contact_sensor_left")
    #     }
    # )

    # # 右腳接觸獎勵
    # contact_ground_right_foot = RewTerm(
    #     func=mdp.contact_ground_reward, 
    #     weight=50.0, 
    #     params={
    #         "threshold": 1.0,
    #         "sensor_cfg": SceneEntityCfg("contact_sensor_right")
    #     }
    # )

    # # 腿部自碰撞懲罰（簡單版本）
    # leg_self_collision_penalty = RewTerm(
    #     func=mdp.leg_self_collision_penalty,
    #     weight=-100.0,  # 負權重表示懲罰
    #     params={
    #         "dangerous_knee_angle": 2.5,      # 約143度
    #         "dangerous_hip_yaw_angle": 0.7    # 約40度
    #     }
    # )
    # ------------------------------------------------------------------------------

    # -- task
    track_lin_vel_xy_exp = RewTerm( #original weight = 1
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # alive = RewTerm(func=mdp.is_alive, weight=10)
    # isend = RewTerm(func=mdp.is_terminated, weight=-100.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-3)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=	-1e-1)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.125,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )
    # # -- optional penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


    # 新增：腳部馬達運動獎勵
    # foot_motor_movement_reward = RewTerm(
    #     func=mdp.foot_motor_movement_reward,
    #     weight=0.5  # 獎勵權重，可以調整
    # )
    
    # # 新增：對稱步態獎勵（選擇其中一個）
    # knee_symmetry_reward = RewTerm(
    #     func=mdp.knee_symmetry_reward,
    #     weight=1.0  # 或使用 simple_knee_alternation_reward
    # )
    
    # 替換為大腿對稱獎勵
    # thigh_symmetry_reward = RewTerm(
    #     func=mdp.thigh_symmetry_reward,
    #     weight=1.0
    # )
    action_smoothness_reward = RewTerm(func=mdp.action_smoothness_reward,weight=5,params={"smoothness_weight": 1.0, "max_change_rate": 0.3})

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.26})



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
