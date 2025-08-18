# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
General.
"""


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
Root penalties.
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
Joint penalties.
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)


"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)

# 測試-----------------------------------------------------------------------------
# def move_towards_target(
#     env: ManagerBasedRLEnv, 
#     target_pos: tuple[float, float, float],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward for moving towards the target position."""
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
    
#     # 將目標位置轉換為張量
#     target_pos_tensor = torch.tensor(target_pos, device=env.device, dtype=torch.float32)
    
#     # 獲取機器人當前位置和速度
#     current_pos = asset.data.root_pos_w[:, :3]  # 取 x, y, z 坐標
#     current_velocity = asset.data.root_lin_vel_b[:, :2]  # 取 x, y 速度（身體坐標系）
    
#     # 計算到目標的方向向量（只考慮水平方向）
#     to_target_vector = target_pos_tensor[:2] - current_pos[:, :2]  # 忽略 z 軸
#     target_distance = torch.norm(to_target_vector, dim=1, keepdim=True)
    
#     # 計算目標方向的單位向量（避免除零）
#     target_direction = torch.where(
#         target_distance > 0.01,
#         to_target_vector / target_distance,
#         torch.zeros_like(to_target_vector)
#     )
    
#     # 計算速度在目標方向上的投影（點積）
#     velocity_projection = torch.sum(current_velocity * target_direction, dim=1)
    
#     # 只獎勵正向移動（朝向目標），使用 clamp 限制在 0 以上
#     return torch.clamp(velocity_projection, min=0.0)

# # contact sensor rewards
# def contact_ground_reward(
#     env: ManagerBasedRLEnv, 
#     threshold: float = 1.0,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
# ) -> torch.Tensor:
#     """Reward for contact sensor touching the ground."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
#     # 獲取接觸力數據
#     net_contact_forces = contact_sensor.data.net_forces_w_history
    
#     print(f"net_contact_forces shape: {net_contact_forces.shape}")
#     print(f"env.num_envs: {env.num_envs}")
#     print(f"sensor_cfg.body_ids: {sensor_cfg.body_ids}")

#     # 如果沒有指定 body_ids，使用所有 body
#     if hasattr(sensor_cfg, 'body_ids') and sensor_cfg.body_ids is not None:
#         contact_forces = net_contact_forces[:, :, sensor_cfg.body_ids]
#     else:
#         contact_forces = net_contact_forces
    
#     # 計算接觸力的大小
#     contact_force_magnitude = torch.norm(contact_forces, dim=-1)
    
#     # 檢查是否有接觸（對所有時間步和body取最大值）
#     if contact_force_magnitude.dim() > 1:
#         # 如果有多個維度，沿著最後的維度取最大值
#         is_contact = torch.max(contact_force_magnitude.view(env.num_envs, -1), dim=1)[0] > threshold
#     else:
#         is_contact = contact_force_magnitude > threshold
    
#     # 返回獎勵
#     return is_contact.float()

# def leg_self_collision_penalty(
#     env: ManagerBasedRLEnv,
#     dangerous_knee_angle: float = 2.5,
#     dangerous_hip_yaw_angle: float = 0.7,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize leg self-collision when hip_yaw_link contacts knee_link based on joint angles."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
    
#     # 獲取關節位置和名稱
#     joint_pos = asset.data.joint_pos  # [num_envs, num_joints]
#     joint_names = asset.joint_names
    
#     # 添加調試輸出 - 每100步打印一次
#     if hasattr(env, 'common_step_counter'):
#         if env.common_step_counter % 100 == 0:
#             print(f"Step {env.common_step_counter}:")
#             print(f"Available joint names: {joint_names}")
#             print(f"Joint positions shape: {joint_pos.shape}")
#             # 打印第一個環境的所有關節角度
#             if env.num_envs > 0:
#                 print(f"Environment 0 joint angles:")
#                 for i, name in enumerate(joint_names):
#                     angle_deg = torch.rad2deg(joint_pos[0, i]).item()
#                     print(f"  {name}: {joint_pos[0, i]:.4f} rad ({angle_deg:.2f}°)")
    
#     # 找出相關關節的索引
#     joint_indices = {
#         'left_hip_yaw': None,
#         'right_hip_yaw': None,
#         'left_knee': None,
#         'right_knee': None
#     }
    
#     for i, name in enumerate(joint_names):
#         name_lower = name.lower()
#         if "left" in name_lower and "hip" in name_lower and "yaw" in name_lower:
#             joint_indices['left_hip_yaw'] = i
#         elif "right" in name_lower and "hip" in name_lower and "yaw" in name_lower:
#             joint_indices['right_hip_yaw'] = i
#         elif "left" in name_lower and "knee" in name_lower:
#             joint_indices['left_knee'] = i
#         elif "right" in name_lower and "knee" in name_lower:
#             joint_indices['right_knee'] = i
    
#     # 打印找到的關節索引（僅在第一次調用時）
#     if not hasattr(env, '_joint_indices_printed'):
#         print(f"Found joint indices: {joint_indices}")
#         env._joint_indices_printed = True
    
#     # 初始化懲罰
#     penalty = torch.zeros(env.num_envs, device=env.device)
    
#     # 檢查左腿自碰撞風險
#     if joint_indices['left_hip_yaw'] is not None and joint_indices['left_knee'] is not None:
#         left_hip_yaw = joint_pos[:, joint_indices['left_hip_yaw']]
#         left_knee = joint_pos[:, joint_indices['left_knee']]
        
#         # 添加關鍵角度的調試輸出
#         if hasattr(env, 'common_step_counter') and env.common_step_counter % 500 == 0:
#             print(f"Left leg angles - Hip yaw: {torch.rad2deg(left_hip_yaw[0]):.2f}°, Knee: {torch.rad2deg(left_knee[0]):.2f}°")
        
#         knee_dangerous = torch.abs(left_knee) > dangerous_knee_angle
#         hip_yaw_dangerous = torch.abs(left_hip_yaw) > dangerous_hip_yaw_angle
        
#         left_collision_risk = knee_dangerous & hip_yaw_dangerous
        
#         # 當檢測到碰撞風險時立即打印
#         if left_collision_risk.any():
#             collision_envs = torch.where(left_collision_risk)[0]
#             for env_idx in collision_envs[:3]:  # 只打印前3個環境避免輸出過多
#                 print(f"LEFT LEG COLLISION RISK in env {env_idx}: Hip yaw={torch.rad2deg(left_hip_yaw[env_idx]):.2f}°, Knee={torch.rad2deg(left_knee[env_idx]):.2f}°")
        
#         penalty += left_collision_risk.float()
    
#     # 檢查右腿自碰撞風險
#     if joint_indices['right_hip_yaw'] is not None and joint_indices['right_knee'] is not None:
#         right_hip_yaw = joint_pos[:, joint_indices['right_hip_yaw']]
#         right_knee = joint_pos[:, joint_indices['right_knee']]
        
#         # 添加關鍵角度的調試輸出
#         if hasattr(env, 'common_step_counter') and env.common_step_counter % 500 == 0:
#             print(f"Right leg angles - Hip yaw: {torch.rad2deg(right_hip_yaw[0]):.2f}°, Knee: {torch.rad2deg(right_knee[0]):.2f}°")
        
#         knee_dangerous = torch.abs(right_knee) > dangerous_knee_angle
#         hip_yaw_dangerous = torch.abs(right_hip_yaw) > dangerous_hip_yaw_angle
        
#         right_collision_risk = knee_dangerous & hip_yaw_dangerous
        
#         # 當檢測到碰撞風險時立即打印
#         if right_collision_risk.any():
#             collision_envs = torch.where(right_collision_risk)[0]
#             for env_idx in collision_envs[:3]:
#                 print(f"RIGHT LEG COLLISION RISK in env {env_idx}: Hip yaw={torch.rad2deg(right_hip_yaw[env_idx]):.2f}°, Knee={torch.rad2deg(right_knee[env_idx]):.2f}°")
        
#         penalty += right_collision_risk.float()
    
#     return penalty


# def leg_extreme_angles_penalty(
#     env: ManagerBasedRLEnv,
#     max_knee_flex: float = 2.8,  # 最大膝關節彎曲角度（約160度）
#     max_hip_yaw: float = 0.8,    # 最大髖關節偏轉角度（約45度）
#     combination_threshold: float = 0.7,  # 組合危險閾值係數
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize extreme leg angles that may cause self-collision."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
    
#     joint_pos = asset.data.joint_pos
#     joint_names = asset.joint_names
    
#     # 找出關節索引
#     joint_indices = {}
#     for i, name in enumerate(joint_names):
#         name_lower = name.lower()
#         if "left" in name_lower and "hip" in name_lower and "yaw" in name_lower:
#             joint_indices['left_hip_yaw'] = i
#         elif "right" in name_lower and "hip" in name_lower and "yaw" in name_lower:
#             joint_indices['right_hip_yaw'] = i
#         elif "left" in name_lower and "knee" in name_lower:
#             joint_indices['left_knee'] = i
#         elif "right" in name_lower and "knee" in name_lower:
#             joint_indices['right_knee'] = i
    
#     penalty = torch.zeros(env.num_envs, device=env.device)
    
#     # 檢查左腿
#     if joint_indices.get('left_hip_yaw') is not None and joint_indices.get('left_knee') is not None:
#         left_hip_yaw = joint_pos[:, joint_indices['left_hip_yaw']]
#         left_knee = joint_pos[:, joint_indices['left_knee']]
        
#         # 計算危險程度（0-1之間）
#         knee_danger = torch.clamp(torch.abs(left_knee) / max_knee_flex, 0, 1)
#         hip_danger = torch.clamp(torch.abs(left_hip_yaw) / max_hip_yaw, 0, 1)
        
#         # 組合危險度：當兩個角度都達到一定程度時懲罰
#         combined_danger = knee_danger * hip_danger
#         left_collision = combined_danger > combination_threshold
        
#         penalty += left_collision.float()
    
#     # 檢查右腿
#     if joint_indices.get('right_hip_yaw') is not None and joint_indices.get('right_knee') is not None:
#         right_hip_yaw = joint_pos[:, joint_indices['right_hip_yaw']]
#         right_knee = joint_pos[:, joint_indices['right_knee']]
        
#         # 計算危險程度
#         knee_danger = torch.clamp(torch.abs(right_knee) / max_knee_flex, 0, 1)
#         hip_danger = torch.clamp(torch.abs(right_hip_yaw) / max_hip_yaw, 0, 1)
        
#         # 組合危險度
#         combined_danger = knee_danger * hip_danger
#         right_collision = combined_danger > combination_threshold
        
#         penalty += right_collision.float()
    
#     return penalty
def foot_motor_movement_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward when foot motors are moving (have joint velocity)."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    joint_names = asset.joint_names
    joint_vel = asset.data.joint_vel  # 關節速度
    
    # 找出腳部相關關節的索引
    foot_joint_indices = []
    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        # 尋找包含 ankle 或 foot 的關節
        if 'ankle' in name_lower or 'foot' in name_lower:
            foot_joint_indices.append(i)
    
    # 如果沒找到腳部關節，返回零獎勵
    if not foot_joint_indices:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 使用 tanh 函數將獎勵限制在 0-1 之間
    reward = torch.tanh(torch.sum(torch.abs(joint_vel[:, foot_joint_indices]), dim=1))
    
    return reward