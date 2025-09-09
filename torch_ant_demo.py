import torch
import torch.nn as nn
import numpy as np

# import the skrl components
from skrl.agents.torch.sac import SAC
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# ========== 定義 Actor 與 Critic ==========
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions),
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
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

# ========== 載入環境 ==========
env = load_isaaclab_env(task_name="Isaac-Velocity-Flat-G1-v0", num_envs=1)
env = wrap_env(env)
device = env.device

# ========== 建立 agent ==========
models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

agent = SAC(models=models,
            memory=None,
            cfg={},   # 測試時不需要配置
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# ========== 載入已訓練好的模型 ==========
agent.load("/home/fcuai/IsaacLab/runs/torch/Isaac-Ant-v0/25-08-28_19-50-27-963494_SAC/checkpoints/best_agent.pt")

# ========== 展示迴圈 ==========
obs, _ = env.reset()
for step in range(2000):
    with torch.no_grad():
        # act 可能回傳 tuple: (actions, log_prob, info)
        result = agent.act(obs, timestep=step, timesteps=2000)

        # 如果回傳的是 tuple，取第一個 (actions)
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result  # 已經是 tensor

    obs, reward, terminated, truncated, info = env.step(action)

    if np.any(terminated) or np.any(truncated):
        obs, _ = env.reset()

env.close()
