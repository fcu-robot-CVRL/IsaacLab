import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
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


# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Velocity-Flat-G1-v0", num_envs=64)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models = {}
models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.3, base_scale=4.0, device=device)#sigma0.0base2.0
cfg["gradient_steps"] = 1
cfg["batch_size"] = 1024
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.023
cfg["actor_learning_rate"] = 2e-5
cfg["critic_learning_rate"] = 2e-4
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["replay_buffer_size"] = 2e6

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/torch/Isaac-Ant-v0"

agent = DDPG(models=models,
             memory=memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)

# agent.load("/media/fcuai/KINGSTON/Isaac-Ant-v0/25-09-17_11-47-18-229223_DDPG/checkpoints/best_agent.pt")
# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
