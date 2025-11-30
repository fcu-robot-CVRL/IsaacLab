import torch
import torch.nn as nn
import copy

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import ParallelTrainer,SequentialTrainer
from skrl.utils import set_seed
import gym
import numpy as np
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise


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

class DeterministicActor_DDPG(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

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

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic_DDPG(DeterministicMixin, Model):
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
# env = load_isaaclab_env(task_name="Isaac-Velocity-Flat-G1-v0", num_envs=8)
env = load_isaaclab_env(task_name="Isaac-Velocity-Flat-G1-v0", num_envs=8)
env_unwrapped = env.unwrapped  # Access the unwrapped environment

# 獲取機器人實例並設置初始狀態
robot = env_unwrapped.scene.articulations["robot"]

spacing = 2.5  # 機器人之間的間距
for i in range(env_unwrapped.num_envs):
    # print(robot.data.default_root_state[i, :3])
    x_offset = robot.data.root_pos_w[0, 0] - robot.data.root_pos_w[i, 0] - i * spacing  # 排成一排
    y_offset = robot.data.root_pos_w[0, 1] - robot.data.root_pos_w[i, 1]
    
    robot.data.default_root_state[i, 0] = x_offset
    robot.data.default_root_state[i, 1] = y_offset
    # print(robot.data.default_root_state[i, 0:3])
# 將修改後的初始狀態寫回模擬器
robot.write_root_state_to_sim(robot.data.default_root_state)


action_term = env_unwrapped.action_manager.get_term("joint_pos")
original_scale = action_term.cfg.scale
scale_per_robot = torch.tensor(
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25], 
    device=env_unwrapped.device,
    dtype=torch.float32
)
robot_info = [
    ("Robot 0", "SAC-1", "agent"),
    ("Robot 1", "SAC-2", "agent2"),
    ("Robot 2", "SAC-3", "agent3"),
    ("Robot 3", "SAC-4", "agent4"),
    ("Robot 4", "SAC-5", "agent5"),
    ("Robot 5", "SAC-6", "agent6"),
    ("Robot 6", "DDPG-1", "agent7"),
    ("Robot 7", "DDPG-2", "agent8"),
]
for i, (robot_name, algo_name, var_name) in enumerate(robot_info):
    scale_value = scale_per_robot[i].item()
    marker = "🔥" if scale_value != original_scale else "  "
original_process_actions = action_term.process_actions
scale_multiplier = scale_per_robot / original_scale
def custom_process_actions(actions):
    scaled_actions = actions * scale_multiplier.view(-1, 1)
    return original_process_actions(scaled_actions)
action_term.process_actions = custom_process_actions

env = wrap_env(env_unwrapped)
# env = SafeActionWrapper(env)  # optional: wrap the environment to ensure safe actions


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

models_DDPG = {}
models_DDPG["policy"] = DeterministicActor_DDPG(env.observation_space, env.action_space, device)
models_DDPG["target_policy"] = DeterministicActor_DDPG(env.observation_space, env.action_space, device)
models_DDPG["critic"] = Critic_DDPG(env.observation_space, env.action_space, device)
models_DDPG["target_critic"] = Critic_DDPG(env.observation_space, env.action_space, device)
models_DDPG2 = copy.deepcopy(models_DDPG)
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
cfg_DDPG = DDPG_DEFAULT_CONFIG.copy()
cfg_DDPG["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.3, base_scale=0.001, device=device)#sigma0.0base2.0
cfg_DDPG["gradient_steps"] = 1
cfg_DDPG["batch_size"] = 1024
cfg_DDPG["discount_factor"] = 0.99
cfg_DDPG["polyak"] = 0.023
cfg_DDPG["actor_learning_rate"] = 2e-5
cfg_DDPG["critic_learning_rate"] = 2e-4
cfg_DDPG["random_timesteps"] = 0
cfg_DDPG["learning_starts"] = 0
cfg_DDPG["state_preprocessor"] = RunningStandardScaler
cfg_DDPG["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_DDPG["replay_buffer_size"] = 2e6

# logging to TensorBoard and write checkpoints (in timesteps)
cfg_DDPG["experiment"]["write_interval"] = 800
cfg_DDPG["experiment"]["checkpoint_interval"] = 8000
cfg_DDPG["experiment"]["directory"] = "runs/torch/Isaac-Ant-v0"
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
agent7 = DDPG(models=models_DDPG,
             memory=memory7,
             cfg=cfg_DDPG,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)
            # cfg=cfg,
            # observation_space=env.observation_space,
            # action_space=env.action_space,
            # device=device)
agent8 = DDPG(models=models_DDPG2,
             memory=memory8,
             cfg=cfg_DDPG,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)
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
# agent7.load("/home/fcuai/IsaacLab/runs/torch/Isaac-Ant-v0/25-10-29_01-16-57-916063_DDPG/checkpoints/best_agent.pt")
# agent8.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-09-17_11-47-18-229223_DDPG/checkpoints/best_agent.pt")
agent7.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-11-03_20-23-54-438451_DDPG/checkpoints/best_agent.pt")
agent8.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-11-04_21-37-40-245396_DDPG/checkpoints/best_agent.pt")
# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": False}
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent, agent2],agents_scope=[1,1])


trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent, agent2, agent3, agent4, agent5, agent6, agent7, agent8], agents_scope=[1,1,1,1,1,1,1,1])
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent, agent2], agents_scope=[1,1])

# start
trainer.eval()
