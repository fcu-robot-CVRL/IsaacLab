# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # self.scene.imu_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.imu_scanner_pelvis.prim_path = "{ENV_REGEX_NS}/Robot/g1/waist_link"
        self.scene.imu_scanner_L_elbow.prim_path = "{ENV_REGEX_NS}/Robot/g1/left_arm_link"
        self.scene.imu_scanner_R_elbow.prim_path = "{ENV_REGEX_NS}/Robot/g1/right_arm_link"
        self.scene.imu_scanner_L_knee.prim_path = "{ENV_REGEX_NS}/Robot/g1/left_calf_link"
        self.scene.imu_scanner_R_knee.prim_path = "{ENV_REGEX_NS}/Robot/g1/right_calf_link"
        self.scene.contact_sensor_left.prim_path = "{ENV_REGEX_NS}/Robot/g1/left_foot_link"
        self.scene.contact_sensor_right.prim_path = "{ENV_REGEX_NS}/Robot/g1/right_foot_link"
        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight =-0.008# -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.9#0.75
        self.rewards.feet_air_time.params["threshold"] = 0.9#0.4
        self.rewards.dof_torques_l2.weight = -8.0e-8
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_ankle_joint",".*_calf_joint"]
        )
        self.rewards.dof_torques_l2_2.weight = -1.0e-7
        self.rewards.dof_torques_l2_2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_thigh_joint"]
        )
        # 測試-----------------------------------------
        # self.rewards.contact_ground_left_foot.weight = 10.0
        # self.rewards.contact_ground_right_foot.weight = 10.0
        # self.rewards.move_towards_target.weight = 5.0
        # self.rewards.leg_self_collision_penalty.weight = -10.0
        # ---------------------------------------------
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)#(-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)#(-1.0, 1.0)


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
