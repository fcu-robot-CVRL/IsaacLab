# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=None,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                'left_hip_yaw_joint' : 0. ,   
                'left_hip_roll_joint' : 0,               
                'left_hip_pitch_joint' : -0.1,         
                'left_knee_joint' : 0.3,       
                'left_ankle_pitch_joint' : -0.2,     
                'left_ankle_roll_joint' : 0,     
                'right_hip_yaw_joint' : 0., 
                'right_hip_roll_joint' : 0, 
                'right_hip_pitch_joint' : -0.1,                                       
                'right_knee_joint' : 0.3,                                             
                'right_ankle_pitch_joint': -0.2,                              
                'right_ankle_roll_joint' : 0,       
                'torso_joint' : 0.
            }
         ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[
                    # '.*hip_yaw.*',
                    #  '.*hip_roll.*',
                    #  '.*hip_pitch.*',
                    #  '.*knee.*',
                    #  '.*ankle.*'],
                    '.*'],
                stiffness={
                    '.*hip_yaw.*': 100,
                     '.*hip_roll.*': 100,
                     '.*hip_pitch.*': 100,
                     '.*knee.*': 150,
                     '.*ankle.*': 40,
                     '.*':50,  
                },
                damping={
                    '.*hip_yaw.*': 2,
                     '.*hip_roll.*': 2,
                     '.*hip_pitch.*': 2,
                     '.*knee.*': 4,
                     '.*ankle.*': 2,
                     '.*': 2,   
                },
            ),
        },
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*": 0.25,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        """        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)"""
        # base_height = ObsTerm(func=mdp.base_pos_z)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        # base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        # base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (100.0, 0.0, 0.0)})
        # base_up_proj = ObsTerm(func=mdp.base_up_proj)
        # base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (100.0, 0.0, 0.0)})
        # joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        # actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)  # 線性速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0)  # 角速度
        projected_gravity = ObsTerm(func=mdp.projected_gravity)   # 投影重力
        
        commands = ObsTerm(func=mdp.commands, scale=1.0)  # 命令（前3項）
        dof_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)      # 關節位置（相對於默認位置的歸一化）
        dof_vel = ObsTerm(func=mdp.joint_vel, scale=1.0)          # 關節速度
        actions = ObsTerm(func=mdp.last_action)                  # 上一個動作
        # sin_phase = ObsTerm(func=mdp.sin_phase)                  # 相位正弦值
        # cos_phase = ObsTerm(func=mdp.cos_phase)                  # 相位餘弦值
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress = RewTerm(func=mdp.progress_reward, weight=10.0, params={"target_pos": (100.0, 0.0, 0.0)})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # (3) Reward for non-upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    # (4) Reward for moving in the right direction
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=5, params={"threshold": 0.8, "target_pos": (100.0, 0.0, 0.0)}
    )
    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # lessthen = RewTerm(func=mdp.low_max_height_reward, weight=10)
    domove = RewTerm(func=mdp.movement_activity_reward, weight=10)
    # (6) Penalty for energy consumption
    # energy = RewTerm(
    #     func=mdp.power_consumption,
    #     weight=-0.005,
    #     params={
    #         "gear_ratio": {
    #             ".*_waist.*": 67.5,
    #             ".*_upper_arm.*": 67.5,
    #             "pelvis": 67.5,
    #             ".*_lower_arm": 45.0,
    #             ".*_thigh:0": 45.0,
    #             ".*_thigh:1": 135.0,
    #             ".*_thigh:2": 45.0,
    #             ".*_shin": 90.0,
    #             ".*_foot.*": 22.5,
    #         }
    #     },
    # )
    # (7) Penalty for reaching close to joint limits
    # joint_pos_limits = RewTerm(
    #     func=mdp.joint_pos_limits_penalty_ratio,
    #     weight=-0.25,
    #     params={
    #         "threshold": 0.98,
    #         "gear_ratio": {
    #             ".*_waist.*": 67.5,
    #             ".*_upper_arm.*": 67.5,
    #             "pelvis": 67.5,
    #             ".*_lower_arm": 45.0,
    #             ".*_thigh:0": 45.0,
    #             ".*_thigh:1": 135.0,
    #             ".*_thigh:2": 45.0,
    #             ".*_shin": 90.0,
    #             ".*_foot.*": 22.5,
    #         },
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.35})


@configclass
class HumanoidEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
