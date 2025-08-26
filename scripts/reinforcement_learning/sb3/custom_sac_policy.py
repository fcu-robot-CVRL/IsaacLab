import torch
import torch.nn as nn
from stable_baselines3.sac.policies import SACPolicy, Actor


# 你自訂的 Actor，會被 SACPolicy 用到
class ReLUClampedActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 改寫輸出層為 ReLU + Tanh 限制在 [-1, 1]
        self.mu = nn.Sequential(
            nn.Linear(self.latent_dim_pi, self.action_dim),
            nn.ReLU(),
            nn.Tanh()
        )

    def _predict(self, latent_pi: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.mu(latent_pi)


# 自訂 SACPolicy，只改 make_actor 就好！
class MySACPolicy(SACPolicy):
    def make_actor(self) -> Actor:
        return ReLUClampedActor(
            self.observation_space,
            self.action_space,
            self.features_extractor,
            self.features_dim,
            self.actor_hidden_dim,
            self.log_std_init,
            self.share_features_extractor,
        ).to(self.device)
