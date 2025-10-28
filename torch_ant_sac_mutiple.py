import torch
import torch.nn as nn
import copy

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import ParallelTrainer,SequentialTrainer
from skrl.utils import set_seed
import gym
import numpy as np


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations,512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),    
                                 nn.ReLU(), 
                                 nn.Linear(64, 32),    
                                 nn.ReLU(),                             
                                 nn.Linear(32, self.num_actions),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),    
                                 nn.ReLU(),
                                 nn.Linear(64, 32),    
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

# class SafeActionWrapper(gym.Wrapper):
#     def step(self, action):
#         # 防止 NaN / INF
#         action = torch.nan_to_num(action, nan=0, posinf=7.5, neginf=-7.5)
#         action = torch.clamp(action, -7.5, 7.5)
#         action = action * 7.5
#         # print(f"Action after clipping: {action}")
#         return self.env.step(action)
    
# load and wrap the Isaac Lab environment
# env = load_isaaclab_env(task_name="Isaac-Velocity-Flat-G1-v0", num_envs=64)
env = load_isaaclab_env(task_name="Isaac-Velocity-Flat-G1-v0", num_envs=6)
env = wrap_env(env)
# env = SafeActionWrapper(env)  # optional: wrap the environment to ensure safe actions

def create_text_markers(env):
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    """創建文字編號標記"""
    
    # 為每個環境創建不同的文字標記
    markers_dict = {}
    
    for i in range(env.unwrapped.num_envs):
        # 創建文字平面標記
        markers_dict[f"text_{i+1}"] = sim_utils.CuboidCfg(
            size=(0.3, 0.1, 0.3),  # 扁平的立方體作為文字背景
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0) if i == 0 else (0.0, 1.0, 1.0),  # 黃色或青色
                metallic=0.0,
                roughness=0.2,
            ),
        )
    
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/TextMarkers",
        markers=markers_dict
    )
    
    return VisualizationMarkers(marker_cfg)


device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=15625, num_envs=1, device=device)
memory2 = RandomMemory(memory_size=15625, num_envs=1, device=device)
memory3 = RandomMemory(memory_size=15625, num_envs=1, device=device)
memory4 = RandomMemory(memory_size=15625, num_envs=1, device=device)
memory5 = RandomMemory(memory_size=15625, num_envs=1, device=device)
memory6 = RandomMemory(memory_size=15625, num_envs=1, device=device)
memory7 = RandomMemory(memory_size=15625, num_envs=1, device=device)
memory8 = RandomMemory(memory_size=15625, num_envs=1, device=device)

# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

models2 = copy.deepcopy(models)
models3 = copy.deepcopy(models)
models4 = copy.deepcopy(models)
models5 = copy.deepcopy(models)
models6 = copy.deepcopy(models)
models7 = copy.deepcopy(models)
models8 = copy.deepcopy(models)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 2048 #4096
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 1e-4 #5e-4
cfg["critic_learning_rate"] = 1e-4 #5e-4
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 8e-4 #1e-3 #5e-3
cfg["initial_entropy_value"] = 0.0001#
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/torch/Isaac-Ant-v0"
cfg2 = copy.deepcopy(cfg)
agent = SAC(models=models,
            memory=memory,
            device=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            cfg=cfg)
            # ,
            # observation_space=env.observation_space,
            # action_space=env.action_space,
            # device=device)
agent2 = SAC(models=models2,
            memory=memory2,
            device=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            cfg=cfg2)
            # cfg=cfg,
            # observation_space=env.observation_space,
            # action_space=env.action_space,
            # device=device)
agent3 = SAC(models=models3,
            memory=memory3,
            device=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            cfg=cfg2)
            # cfg=cfg,
            # observation_space=env.observation_space,
            # action_space=env.action_space,
            # device=device)
agent4 = SAC(models=models4,
            memory=memory4,
            device=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            cfg=cfg2)
            # cfg=cfg,
            # observation_space=env.observation_space,
            # action_space=env.action_space,
            # device=device)
agent5 = SAC(models=models5,
            memory=memory5,
            device=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            cfg=cfg2)
            # cfg=cfg,
            # observation_space=env.observation_space,
            # action_space=env.action_space,
            # device=device)
agent6 = SAC(models=models6,
            memory=memory6,
            device=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            cfg=cfg2)
            # cfg=cfg,
            # observation_space=env.observation_space,
            # action_space=env.action_space,
            # device=device)
# agent7 = SAC(models=models6,
#             memory=memory6,
#             device=device,
#             observation_space=env.observation_space,
#             action_space=env.action_space,
#             cfg=cfg2)
#             # cfg=cfg,
#             # observation_space=env.observation_space,
#             # action_space=env.action_space,
#             # device=device)
# agent8 = SAC(models=models6,
#             memory=memory6,
#             device=device,
#             observation_space=env.observation_space,
#             action_space=env.action_space,
#             cfg=cfg2)
            # cfg=cfg,
            # observation_space=env.observation_space,
            # action_space=env.action_space,
            # device=device)

agent.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-08-29_17-00-48-414720_SAC/checkpoints/best_agent.pt")
agent2.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-09-03_11-03-39-651957_SAC/checkpoints/best_agent.pt")  # optional: load pre-trained agent. Adjust the path as needed.
agent3.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-09-05_12-20-25-663631_SAC/checkpoints/best_agent.pt")
agent4.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-09-05_12-20-25-663631_SAC/checkpoints/best_agent.pt")
agent5.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-09-10_15-19-24-182667_SAC/checkpoints/best_agent.pt")
agent6.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-09-10_16-38-47-649660_SAC/checkpoints/best_agent.pt")
# agent7.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-09-14_11-38-48-144776_DDPG/checkpoints/best_agent.pt")
# agent8.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-09-17_11-47-18-229223_DDPG/checkpoints/best_agent.pt")
# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": False}
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent, agent2],agents_scope=[1,1])

robot_markers = create_text_markers(env)

# 訓練開始前初始化標記位置
def initialize_markers():
    try:
        robot = env.unwrapped.scene["robot"]
        
        # 直接使用根部位置（最安全的方法）
        root_pos = robot.data.root_pos_w.cpu().numpy()
        print(f"根部位置形狀: {root_pos.shape}")
        
        # 確保是正確的形狀 (num_envs, 3)
        if len(root_pos.shape) == 2 and root_pos.shape[1] >= 3:
            marker_positions = root_pos[:, :3].copy()
        elif len(root_pos.shape) == 1 and len(root_pos) >= 3:
            marker_positions = root_pos[:3].reshape(1, 3)
        else:
            # 創建默認位置
            num_envs = env.unwrapped.num_envs
            marker_positions = np.array([[0, 0, 1], [2, 0, 1]])[:num_envs]
        
        # 向上偏移
        marker_positions[:, 2] += 1.0  # 向上1米
        
        # 設置標記
        robot_markers.visualize(
            translations=marker_positions,
            marker_indices=list(range(len(marker_positions)))
        )
        
        print(f"✓ 標記設置成功: {marker_positions}")
        
    except Exception as e:
        print(f"✗ 標記設置失敗: {e}")

def update_markers():
    """實時更新標記位置"""
    try:
        robot = env.unwrapped.scene["robot"]
        
        # 獲取腰部位置（使用之前找到的身體部位）
        if hasattr(update_markers, 'waist_body_id'):
            waist_positions = robot.data.body_pos_w[:, update_markers.waist_body_id, :3].cpu().numpy()
        else:
            # 第一次執行，找到腰部ID
            waist_body_ids = robot.find_bodies(["waist_link"])
            if len(waist_body_ids) > 0:
                update_markers.waist_body_id = waist_body_ids[0]
                waist_positions = robot.data.body_pos_w[:, waist_body_ids[0], :3].cpu().numpy()
            else:
                waist_positions = robot.data.root_pos_w.cpu().numpy()
        
        # 更新標記位置
        marker_positions = waist_positions.copy()
        marker_positions[:, 2] += 0.4  # 腰部上方 40cm
        
        robot_markers.visualize(
            translations=marker_positions,
            marker_indices=[0, 1]
        )
        
    except Exception as e:
        print(f"標記更新失敗: {e}")

# 在訓練循環中定期調用（每100步或每1000步）
class MarkerUpdatingTrainer(SequentialTrainer):
    def single_agent_step(self, env, agent):
        result = super().single_agent_step(env, agent)
        
        # 每500步更新一次標記
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        if self._step_count % 500 == 0:
            update_markers()
        
        return result

# 在trainer.eval()之前調用
initialize_markers()
trainer = MarkerUpdatingTrainer(cfg=cfg_trainer, env=env, agents=[agent, agent2, agent3, agent4, agent5, agent6], agents_scope=[1,1,1,1,1,1])
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent, agent2], agents_scope=[1,1])

# start training
trainer.eval()
