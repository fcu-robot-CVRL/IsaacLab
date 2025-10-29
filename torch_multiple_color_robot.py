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

# 全局變量來控制標記功能
MARKERS_AVAILABLE = False
VisualizationMarkers = None
VisualizationMarkersCfg = None
sim_utils = None
math_utils = None
ISAAC_NUCLEUS_DIR = None

# 嘗試安全導入標記相關功能
def safe_import_markers():
    global MARKERS_AVAILABLE, VisualizationMarkers, VisualizationMarkersCfg, sim_utils, math_utils, ISAAC_NUCLEUS_DIR
    try:
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        MARKERS_AVAILABLE = True
        print("Markers successfully loaded!")
    except Exception as e:
        print(f"Warning: Markers not available - {e}")
        MARKERS_AVAILABLE = False
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
# 添加標記箭頭所需的導入

# 添加標記箭頭所需的導入
try:
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    MARKERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization markers not available: {e}")
    MARKERS_AVAILABLE = False

# 定義標記箭頭函數
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
def define_markers() -> VisualizationMarkers:
    if not MARKERS_AVAILABLE:
        return None
        
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

# 創建標記包裝器類
class MarkerWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        if MARKERS_AVAILABLE:
            self.visualization_markers = define_markers()
            self.up_dir = torch.tensor([0.0, 0.0, 1.0]).to(env.device)
            self.marker_offset = torch.zeros((env.cfg.scene.num_envs, 3)).to(env.device)
            self.marker_offset[:, -1] = 0.5
            
            # 初始化命令方向（簡化版本，可根據需要調整）
            self.commands = torch.randn((env.cfg.scene.num_envs, 3)).to(env.device)
            self.commands[:, -1] = 0.0
            self.commands = self.commands / torch.linalg.norm(self.commands, dim=1, keepdim=True)
            
            # 計算yaw角度
            self._update_yaws()
        else:
            print("Markers not available - running without visualization")
    
    def _update_yaws(self):
        if not MARKERS_AVAILABLE:
            return
            
        ratio = self.commands[:, 1] / (self.commands[:, 0] + 1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:, 0] * gzero[:, 1]
        minus = lzero[:, 0] * lzero[:, 1]
        offsets = torch.pi * plus - torch.pi * minus
        self.yaws = torch.atan(ratio).reshape(-1, 1) + offsets.reshape(-1, 1)
    
    def _visualize_markers(self):
        if not MARKERS_AVAILABLE or self.visualization_markers is None:
            return
            
        try:
            # 獲取機器人位置和方向
            robot_pos = self.env.scene.articulations["robot"].data.root_pos_w
            robot_quat = self.env.scene.articulations["robot"].data.root_quat_w
            
            # 設置標記位置（機器人上方）
            marker_locations = robot_pos + self.marker_offset
            
            # 前進方向箭頭（跟隨機器人朝向）
            forward_marker_orientations = robot_quat
            
            # 命令方向箭頭
            command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()
            
            # 組合位置和旋轉
            loc = torch.vstack((marker_locations, marker_locations))
            rots = torch.vstack((forward_marker_orientations, command_marker_orientations))
            
            # 標記索引
            all_envs = torch.arange(self.env.cfg.scene.num_envs)
            indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
            
            # 顯示標記
            self.visualization_markers.visualize(loc, rots, marker_indices=indices)
        except Exception as e:
            print(f"Warning: Failed to visualize markers: {e}")
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._visualize_markers()  # 每步更新標記位置
        return obs, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        if MARKERS_AVAILABLE:
            # 重新隨機化命令方向
            self.commands = torch.randn((self.env.cfg.scene.num_envs, 3)).to(self.env.device)
            self.commands[:, -1] = 0.0
            self.commands = self.commands / torch.linalg.norm(self.commands, dim=1, keepdim=True)
            self._update_yaws()
            self._visualize_markers()
        return obs


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

# ...existing code...

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Velocity-Flat-G1-v0", num_envs=6)
env = wrap_env(env)
# 添加標記包裝器
env = MarkerWrapper(env)

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

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent, agent2, agent3, agent4, agent5, agent6], agents_scope=[1,1,1,1,1,1])
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent, agent2], agents_scope=[1,1])

# start
trainer.eval()
