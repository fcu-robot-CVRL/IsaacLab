# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
<<<<<<< HEAD
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_apply
=======

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
>>>>>>> edc12e259a44370c38fd168ba9c51e7ecba9a9ed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

def penalize_y_offset(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), weight: float = -1.0
) -> torch.Tensor:
    """
    Penalize Y軸速度偏移，偏離越多扣分越多。
    Args:
        env: simulation env
        std: 標準差，控制懲罰敏感度
        command_name: 指令名稱
        asset_cfg: 機器人設定
        weight: 懲罰權重（預設負值）
    Returns:
        torch.Tensor: 每個環境的懲罰分數
    """
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    error_y = torch.abs(env.command_manager.get_command(command_name)[:, 1] - vel_yaw[:, 1])
    penalty = weight * torch.square(torch.abs(error_y) / std)
    # print(env.command_manager.get_command(command_name)[:, 1])
    # print(vel_yaw[:, 1])
    # print("Y軸偏移懲罰:", penalty)
    return penalty

def get_body_pos_test(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    body_names = robot.body_names
    body_positions = robot.data.body_pos_w
<<<<<<< HEAD
    left_thigh_link = body_positions[:, body_names.index("left_thigh_link"), :]
    right_thigh_link = body_positions[:, body_names.index("right_thigh_link"), :]
    print("left_thigh_link : ", left_thigh_link[0].cpu().numpy())
    print("right_thigh_link : ", right_thigh_link[0].cpu().numpy())
    # print("所有身體部位位置:")
    # for i, name in enumerate(body_names):
    #     pos = body_positions[0, i].cpu().numpy()  # 取第一個環境
    #     print(f"{i:2d}: {name:25} x={pos[0]:6.3f}, y={pos[1]:6.3f}, z={pos[2]:6.3f}")
    return torch.zeros(env.num_envs, device=env.device)

#me
def leg_dir_ang_diff(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    #legs
    body_names = robot.body_names
    body_positions = robot.data.body_pos_w
    left_thigh_link = body_positions[:, body_names.index("left_thigh_link"), :]
    right_thigh_link = body_positions[:, body_names.index("right_thigh_link"), :]
    #body_dir
    root_quat_w = robot.data.root_link_pose_w[:, 3:7]
    forward_vec_b = robot.data.FORWARD_VEC_B
    forward_vec_w = quat_apply(root_quat_w, forward_vec_b)
    #leg_deg
    leg_vec_xy = right_thigh_link[:, :2] - left_thigh_link[:, :2]
    print("left_thigh_link : ", left_thigh_link[0].cpu().numpy())
    print("right_thigh_link : ", right_thigh_link[0].cpu().numpy())
    print("forward direction (world):", forward_vec_w[0].cpu().numpy())
    print("left->right thigh xy vector:", leg_vec_xy[0].cpu().numpy())
    #θ = Acos( (A dot B) / (|A| |B|) )
    forward_vec_xy = forward_vec_w[:, :2]
    cos_theta = (leg_vec_xy * forward_vec_xy).sum(dim=1) / (leg_vec_xy.norm(dim=1) * forward_vec_xy.norm(dim=1) + 1e-8)
    # 防止數值誤差超過 [-1, 1]
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta_rad = torch.acos(cos_theta)
    theta_deg = theta_rad * 180.0 / torch.pi
    print("angle between leg_vec_xy and forward_vec_w (deg):", theta_deg[0].cpu().item())
    
    # 角度獎勵判斷與計時邏輯
    if not hasattr(env, 'angle_event_buffer'):
        env.angle_event_buffer = [{} for _ in range(env.num_envs)]

    rewards = torch.zeros(env.num_envs, device=env.device)
    current_time = env.episode_length_buf.float() * env.step_dt  # 取得目前時間 (秒)
    for i in range(env.num_envs):
        angle = theta_deg[i].item()
        buffer = env.angle_event_buffer[i]
        # 狀態1：大於120度
        if angle > 120:
            if 'over_120' not in buffer:
                buffer['over_120'] = current_time
            buffer.pop('under_60', None)
        # 狀態2：小於60度
        elif angle < 60:
            if 'under_60' not in buffer:
                buffer['under_60'] = current_time
            buffer.pop('over_120', None)
        # 判斷是否在5秒內完成目標
        # 120->60
        if 'over_120' in buffer and angle < 60:
            if current_time - buffer['over_120'] <= 5.0:
                rewards[i] += 1.0
                buffer.pop('over_120')
        # 60->120
        if 'under_60' in buffer and angle > 120:
            if current_time - buffer['under_60'] <= 5.0:
                rewards[i] += 1.0
                buffer.pop('under_60')

    # 確保回傳型態為 torch.Tensor
    return rewards.clone().to(env.device)
#GPT
# def leg_dir_ang_diff(
#     env, command_name: str, threshold: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward when leg direction angle switches between acute (<45) and obtuse (>135) within 1 second."""
#     robot = env.scene[asset_cfg.name]
#     body_names = robot.body_names
#     body_positions = robot.data.body_pos_w
#     left_thigh_link = body_positions[:, body_names.index("left_thigh_link"), :]
#     right_thigh_link = body_positions[:, body_names.index("right_thigh_link"), :]
#     root_quat_w = robot.data.root_link_pose_w[:, 3:7]
#     forward_vec_b = robot.data.FORWARD_VEC_B
#     forward_vec_w = quat_apply(root_quat_w, forward_vec_b)
#     leg_vec_xy = right_thigh_link[:, :2] - left_thigh_link[:, :2]
#     forward_vec_xy = forward_vec_w[:, :2]
#     cos_theta = (leg_vec_xy * forward_vec_xy).sum(dim=1) / (leg_vec_xy.norm(dim=1) * forward_vec_xy.norm(dim=1) + 1e-8)
#     cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
#     theta_rad = torch.acos(cos_theta)
#     theta_deg = theta_rad * 180.0 / torch.pi

#     # 印出所有主要參數
#     print("left_thigh_link[0]:", left_thigh_link[0].cpu().numpy())
#     print("right_thigh_link[0]:", right_thigh_link[0].cpu().numpy())
#     print("leg_vec_xy[0]:", leg_vec_xy[0].cpu().numpy())
#     print("forward_vec_xy[0]:", forward_vec_xy[0].cpu().numpy())
#     print("cos_theta[0]:", cos_theta[0].cpu().item())
#     print("theta_deg[0]:", theta_deg[0].cpu().item())
#     print("angle_state:", getattr(robot, '_angle_state', None))
#     print("angle_timer:", getattr(robot, '_angle_timer', None))
#     print("acute[0]:", (theta_deg[0] < 45.0).item())
#     print("obtuse[0]:", (theta_deg[0] > 135.0).item())
#     print("reward[0]:", None)  # reward 還沒算

#     # 狀態 buffer
#     if not hasattr(robot, "_angle_state"):
#         robot._angle_state = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)  # 0: 銳角, 1: 鈍角
#         robot._angle_timer = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

#     acute = theta_deg < 60.0
#     obtuse = theta_deg > 120.0
#     reward = torch.zeros(env.num_envs, device=env.device)

#     # 用 torch 操作批次處理
#     mask_acute = (robot._angle_state == 0) & acute
#     robot._angle_timer[mask_acute] += env.step_dt
#     mask_switch_to_obtuse = (robot._angle_state == 0) & obtuse & (robot._angle_timer > 0.0) & (robot._angle_timer <= threshold)
#     reward[mask_switch_to_obtuse] = 1.0
#     robot._angle_state[mask_switch_to_obtuse] = 1
#     robot._angle_timer[mask_switch_to_obtuse] = 0.0
#     mask_reset_acute = (robot._angle_state == 0) & ~(acute | obtuse)
#     robot._angle_timer[mask_reset_acute] = 0.0
#     robot._angle_state[(robot._angle_state == 0) & obtuse] = 1

#     mask_obtuse = (robot._angle_state == 1) & obtuse
#     robot._angle_timer[mask_obtuse] += env.step_dt
#     mask_switch_to_acute = (robot._angle_state == 1) & acute & (robot._angle_timer > 0.0) & (robot._angle_timer <= threshold)
#     reward[mask_switch_to_acute] = 1.0
#     robot._angle_state[mask_switch_to_acute] = 0
#     robot._angle_timer[mask_switch_to_acute] = 0.0
#     mask_reset_obtuse = (robot._angle_state == 1) & ~(acute | obtuse)
#     robot._angle_timer[mask_reset_obtuse] = 0.0
#     robot._angle_state[(robot._angle_state == 1) & acute] = 0

#     mask_init_acute = (robot._angle_state != 0) & acute
#     mask_init_obtuse = (robot._angle_state != 1) & obtuse
#     robot._angle_state[mask_init_acute] = 0
#     robot._angle_state[mask_init_obtuse] = 1
#     robot._angle_timer[mask_init_acute | mask_init_obtuse] = 0.0

#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     print("reward[0]:", reward[0].cpu().item())
#     return reward


def get_forward_direction_test(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    root_quat_w = robot.data.root_link_pose_w[:, 3:7]
    forward_vec_b = robot.data.FORWARD_VEC_B
    forward_vec_w = quat_apply(root_quat_w, forward_vec_b)

    print("forward direction (world):", forward_vec_w.cpu().numpy())
    return torch.zeros(env.num_envs, device=env.device)
=======
    print("所有身體部位位置:")
    for i, name in enumerate(body_names):
        pos = body_positions[0, i].cpu().numpy()  # 取第一個環境
        print(f"{i:2d}: {name:25} x={pos[0]:6.3f}, y={pos[1]:6.3f}, z={pos[2]:6.3f}")
    
    return body_positions
>>>>>>> edc12e259a44370c38fd168ba9c51e7ecba9a9ed

def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def alternating_step_reward(
    env, command_name:str, sensor_cfg: SceneEntityCfg,max_last_time=0.5
) -> torch.Tensor:
    """
    Reward when robot alternates single-leg contact within a given time window.
    
    Args:
        env: simulation env
        sensor_cfg: sensor config
        threshold: reward clamp
        command_name: command name
        window: time window (s) to keep rewarding after a valid swap
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]   # (N, 2)
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids] # (N, 2)

    left_now, right_now = contact_time[:, 0], contact_time[:, 1]
    left_last, right_last = last_contact_time[:, 0], last_contact_time[:, 1]

    # 換腳獎勵：另一腳剛離地，現在這腳接觸，reward = 上一隻腳離地時間
    reward_left = torch.where(
        (left_now > 0.0) & (0.0 < right_last) & (right_last <= max_last_time),
        right_last,
        torch.zeros_like(left_now)
    )

    reward_right = torch.where(
        (right_now > 0.0) & (0.0 < left_last) & (left_last <= max_last_time),
        left_last,
        torch.zeros_like(right_now)
    )

    reward = reward_left + reward_right

    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    return reward