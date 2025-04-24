# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)

class low_max_height_reward(ManagerTermBase):
    """獎勵機器人保持低最高點高度的獎勵項。"""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # 初始化基類
        super().__init__(cfg, env)
        # 從配置中獲取高度閾值，預設為0.3
        self.height_threshold = cfg.params.get("height_threshold", 0.4)
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # 提取機器人資產
        asset: Articulation = env.scene[asset_cfg.name]
        
        # 獲取機器人所有部位的全局位置
        body_positions = asset.data.body_pos_w  # 假設這個屬性包含了所有身體部位的世界坐標
        
        # 找出機器人所有部位中的最高點
        max_height = torch.max(body_positions[:, :, 2], dim=1)[0]
        
        # 計算並返回獎勵
        return torch.where(
            max_height < self.height_threshold,
            torch.ones(env.num_envs, device=env.device),  # 低於閾值時給予滿分獎勵
            torch.exp (-0.1*(max_height - self.height_threshold)) # 高度越高，獎勵越低
        )
        
class movement_activity_reward(ManagerTermBase):
    """獎勵機器人保持活動狀態，防止其站在原地不動。"""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # 初始化基類
        super().__init__(cfg, env)
        # 從配置中獲取參數
        self.min_velocity_threshold = cfg.params.get("min_velocity_threshold", 0.1)  # 最小期望速度
        self.velocity_scale = cfg.params.get("velocity_scale", 1.0)  # 速度獎勵比例係數
        self.joint_movement_scale = cfg.params.get("joint_movement_scale", 0.5)  # 關節運動獎勵比例係數
        
        # 創建歷史記錄
        self.prev_positions = torch.zeros((env.num_envs, 3), device=env.device)
        self.prev_joint_pos = torch.zeros((env.num_envs, env.scene["robot"].num_joints), device=env.device)

    def reset(self, env_ids: torch.Tensor):
        # 重置時更新位置記錄
        asset: Articulation = self._env.scene["robot"]
        self.prev_positions[env_ids] = asset.data.root_pos_w[env_ids, :3]
        self.prev_joint_pos[env_ids] = asset.data.joint_pos[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # 提取機器人資產
        asset: Articulation = env.scene[asset_cfg.name]
        
        # 計算位置變化（線性移動）
        current_positions = asset.data.root_pos_w[:, :3]
        position_delta = torch.norm(current_positions - self.prev_positions, dim=-1)
        
        # 計算關節角度變化
        current_joint_pos = asset.data.joint_pos
        joint_movement = torch.sum(torch.abs(current_joint_pos - self.prev_joint_pos), dim=-1)
        
        # 獲取線速度和角速度的大小
        linear_velocity = torch.norm(asset.data.root_lin_vel_w, dim=-1)
        angular_velocity = torch.norm(asset.data.root_ang_vel_w, dim=-1)
        
        # 更新歷史記錄
        self.prev_positions[:] = current_positions[:]
        self.prev_joint_pos[:] = current_joint_pos[:]
        
        # 計算獎勵
        # 1. 基於速度的獎勵
        velocity_reward = torch.clamp(
            self.velocity_scale * (linear_velocity + 0.5 * angular_velocity), 
            min=0.0, 
            max=1.0
        )
        
        # 2. 基於關節運動的獎勵
        joint_reward = torch.clamp(
            self.joint_movement_scale * joint_movement,
            min=0.0,
            max=1.0
        )
        
        # 組合獎勵，確保有最小值以防止完全不動
        combined_reward = torch.max(velocity_reward, joint_reward)
        
        # 如果完全靜止（速度和關節運動都很小），給予懲罰
        is_static = (linear_velocity < self.min_velocity_threshold) & (joint_movement < 0.01)
        final_reward = torch.where(
            is_static,
            torch.zeros_like(combined_reward) - 0.2,  # 靜止懲罰
            combined_reward
        )
        
        return final_reward
    
class leg_stepping_reward(ManagerTermBase):
    """獎勵機器人通過腿部踏步動作前進，而非僅靠身體傾倒。"""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # 初始化基類
        super().__init__(cfg, env)
        
        # 從配置中獲取參數
        self.leg_joint_names = cfg.params.get("leg_joint_names", ["left_hip", "left_knee", "right_hip", "right_knee"])
        self.stepping_scale = cfg.params.get("stepping_scale", 1.0)
        self.min_leg_movement = cfg.params.get("min_leg_movement", 0.05)
        self.forward_progress_weight = cfg.params.get("forward_progress_weight", 0.7)
        
        # 獲取機器人資產
        asset: Articulation = env.scene["robot"]
        
        # 找出腿部關節索引
        self.leg_joint_indices = []
        for name in self.leg_joint_names:
            indices = [i for i, joint_name in enumerate(asset.joint_names) if name in joint_name]
            self.leg_joint_indices.extend(indices)
        
        # 確保找到了腿部關節
        if not self.leg_joint_indices:
            print("警告: 未找到任何匹配的腿部關節名稱!")
        
        # 初始化歷史數據
        self.prev_joint_pos = torch.zeros((env.num_envs, asset.num_joints), device=env.device)
        self.prev_root_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self.prev_root_rot = torch.zeros((env.num_envs, 4), device=env.device)  # 四元數表示

    def reset(self, env_ids: torch.Tensor):
        # 重置時更新歷史記錄
        asset: Articulation = self._env.scene["robot"]
        self.prev_joint_pos[env_ids] = asset.data.joint_pos[env_ids]
        self.prev_root_pos[env_ids] = asset.data.root_pos_w[env_ids, :3]
        self.prev_root_rot[env_ids] = asset.data.root_quat_w[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # 獲取機器人資產
        asset: Articulation = env.scene[asset_cfg.name]
        
        # 獲取當前狀態
        current_joint_pos = asset.data.joint_pos
        current_root_pos = asset.data.root_pos_w[:, :3]
        current_root_quat = asset.data.root_quat_w
        
        # 計算腿部關節的運動
        leg_joints_movement = torch.sum(
            torch.abs(
                current_joint_pos[:, self.leg_joint_indices] - 
                self.prev_joint_pos[:, self.leg_joint_indices]
            ), 
            dim=1
        )
        
        # 計算前進方向的進展
        # 1. 獲取機器人的前向方向
        forward_dir = torch.zeros((env.num_envs, 3), device=env.device)
        forward_dir[:, 0] = 1.0  # 假設x軸是前方
        # 將前向方向從局部坐標轉換到世界坐標
        forward_dir_world = math_utils.quat_rotate(current_root_quat, forward_dir)
        
        # 2. 計算沿著前向方向的移動距離
        movement_vector = current_root_pos - self.prev_root_pos
        forward_progress = torch.sum(movement_vector * forward_dir_world[:, :3], dim=1)
        
        # 3. 計算身體傾斜度變化（檢測身體傾倒）
        # 獲取向上方向(假設為z軸)在世界座標系中的投影
        up_dir = torch.zeros((env.num_envs, 3), device=env.device)
        up_dir[:, 2] = 1.0  # z軸為上方
        current_up_proj = math_utils.quat_rotate(current_root_quat, up_dir)[:, 2]
        prev_up_proj = math_utils.quat_rotate(self.prev_root_rot, up_dir)[:, 2]
        body_tilt_change = prev_up_proj - current_up_proj  # 正值表示更傾斜
        
        # 獎勵計算:
        # 1. 腿部主動踏步的獎勵
        stepping_reward = torch.clamp(
            self.stepping_scale * leg_joints_movement,
            min=0.0,
            max=1.0
        )
        
        # 2. 懲罰僅靠傾倒而不踏步的行為
        tilt_only_penalty = torch.where(
            (leg_joints_movement < self.min_leg_movement) & (body_tilt_change > 0.02) & (forward_progress > 0.01),
            torch.ones_like(stepping_reward) * -0.5,  # 懲罰
            torch.zeros_like(stepping_reward)
        )
        
        # 3. 獎勵腿部運動與前進結合的行為
        combined_stepping_progress = torch.where(
            (leg_joints_movement > self.min_leg_movement) & (forward_progress > 0.01),
            self.forward_progress_weight * forward_progress + (1 - self.forward_progress_weight) * stepping_reward,
            stepping_reward
        )
        
        # 更新歷史狀態
        self.prev_joint_pos[:] = current_joint_pos[:]
        self.prev_root_pos[:] = current_root_pos[:]
        self.prev_root_rot[:] = current_root_quat[:]
        
        # 返回最終獎勵
        return combined_stepping_progress + tilt_only_penalty