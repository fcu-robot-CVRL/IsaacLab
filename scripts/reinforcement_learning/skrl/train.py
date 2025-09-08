# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="SAC",
    choices=["AMP", "SAC", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime
import torch
import torch.nn as nn

import omni
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
    # import the skrl components to build the RL system for torch
    from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
    from skrl.envs.loaders.torch import load_isaaclab_env
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["sac"] else f"skrl_{algorithm}_cfg_entry_point"


# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        # safely get dimensions
        if hasattr(observation_space, 'shape') and len(observation_space.shape) > 0:
            obs_dim = observation_space.shape[0]
        elif hasattr(observation_space, 'n'):
            obs_dim = observation_space.n
        else:
            obs_dim = self.num_observations if self.num_observations else 1
            
        if hasattr(action_space, 'shape') and len(action_space.shape) > 0:
            action_dim = action_space.shape[0]
        elif hasattr(action_space, 'n'):
            action_dim = action_space.n
        else:
            action_dim = self.num_actions if self.num_actions else 1

        self.net = nn.Sequential(nn.Linear(obs_dim, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, action_dim),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # safely get dimensions
        if hasattr(observation_space, 'shape') and len(observation_space.shape) > 0:
            obs_dim = observation_space.shape[0]
        elif hasattr(observation_space, 'n'):
            obs_dim = observation_space.n
        else:
            obs_dim = self.num_observations if self.num_observations else 1
            
        if hasattr(action_space, 'shape') and len(action_space.shape) > 0:
            action_dim = action_space.shape[0]
        elif hasattr(action_space, 'n'):
            action_dim = action_space.n
        else:
            action_dim = self.num_actions if self.num_actions else 1

        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    print(agent_cfg)
    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)
    
    print(algorithm)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = os.path.join(log_root_path, log_dir)
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # Use custom models for SAC when using torch framework
    if args_cli.ml_framework.startswith("torch") and algorithm == "sac":
        # seed for reproducibility
        set_seed(agent_cfg["seed"])
        
        # get observation and action dimensions from the wrapped environment
        print(f"[INFO] Environment observation space: {env.observation_space}")
        print(f"[INFO] Environment action space: {env.action_space}")
        print(f"[INFO] Environment num_envs: {env.num_envs}")
        print(f"[INFO] Environment device: {env.device}")
        
        # Check if observation_space has proper shape
        if hasattr(env.observation_space, 'shape') and len(env.observation_space.shape) > 0:
            obs_size = env.observation_space.shape[0]
        elif hasattr(env, 'num_observations'):
            obs_size = env.num_observations
        else:
            raise ValueError("Cannot determine observation space size")
            
        if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0:
            action_size = env.action_space.shape[0]
        elif hasattr(env, 'num_actions'):
            action_size = env.num_actions
        else:
            raise ValueError("Cannot determine action space size")
        
        print(f"[INFO] Observation size: {obs_size}, Action size: {action_size}")
        print(f"[INFO] Memory requirement estimate: {(agent_cfg['agent']['memory_size'] * obs_size * 4) / (1024**3):.2f} GB")
        
        # Create a proper observation space for memory initialization
        from gymnasium.spaces import Box
        import numpy as np
        
        # Create proper spaces with correct dimensions
        proper_obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        proper_action_space = Box(low=-np.inf, high=np.inf, shape=(action_size,), dtype=np.float32)
        
        # Clear GPU cache before creating memory
        torch.cuda.empty_cache()
        
        # instantiate a memory as rollout buffer with proper observation space
        try:
            memory = RandomMemory(memory_size=agent_cfg["agent"]["memory_size"], 
                                num_envs=env.num_envs, device=env.device)
        except torch.cuda.OutOfMemoryError:
            print("[ERROR] CUDA out of memory when creating replay buffer.")
            print(f"[INFO] Try reducing memory_size in config (current: {agent_cfg['agent']['memory_size']})")
            print(f"[INFO] Or reduce num_envs (current: {env.num_envs})")
            raise

        # instantiate the agent's models using custom classes with proper spaces
        models = {}
        models["policy"] = StochasticActor(proper_obs_space, proper_action_space, env.device)
        models["critic_1"] = Critic(proper_obs_space, proper_action_space, env.device)
        models["critic_2"] = Critic(proper_obs_space, proper_action_space, env.device)
        models["target_critic_1"] = Critic(proper_obs_space, proper_action_space, env.device)
        models["target_critic_2"] = Critic(proper_obs_space, proper_action_space, env.device)

        # configure and instantiate the agent
        cfg_agent = agent_cfg["agent"].copy()
        cfg_agent.pop("class", None)  # remove class key
        
        # configure SAC with proper settings using proper observation space
        cfg_agent["state_preprocessor"] = RunningStandardScaler
        cfg_agent["state_preprocessor_kwargs"] = {"size": proper_obs_space, "device": env.device}
        
        agent = SAC(models=models,
                   memory=memory,
                   cfg=cfg_agent,
                   observation_space=proper_obs_space,
                   action_space=proper_action_space,
                   device=env.device)

        # configure and instantiate the RL trainer
        trainer_cfg = agent_cfg["trainer"].copy()
        trainer_cfg.pop("class", None)  # remove class key
        trainer_cfg["close_environment_at_exit"] = False
        
        trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
        
        # start training
        trainer.train()

    else:
        # Use default runner for other algorithms or frameworks
        runner = Runner(env, agent_cfg)

        # load checkpoint (if specified)
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)
        print(runner.agent)
        # run training
        runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
