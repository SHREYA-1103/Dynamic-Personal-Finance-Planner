import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from environment.financial_env import FinancialEnv


def make_env(market_df, user_df, config):
    return FinancialEnv(market_df, user_df, config)


# =========================
# CVaR-AWARE CALLBACK (SAFE VERSION)
# =========================
class CVaRCallback(BaseCallback):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []

        self.recent_rewards = []
        self.cvar_values = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        # Track episode return
        self.current_episode_reward += reward

        # Store recent rewards (for threshold estimation)
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 1000:
            self.recent_rewards.pop(0)

        # -------------------------
        # SAFE CVaR BIAS (NO BUFFER HACK)
        # -------------------------
        if len(self.recent_rewards) > 100:
            threshold = np.percentile(self.recent_rewards, self.alpha * 100)

            # Slightly emphasize bad outcomes
            if reward < threshold:
                self.locals["rewards"][0] *= 1.2  # mild bias (safe)

        # Episode end → compute CVaR
        if done:
            self.cvar_values.append(self.current_episode_reward)
            self.current_episode_reward = 0

        return True

    def _on_rollout_end(self) -> None:
        logger_dict = self.model.logger.name_to_value

        if "train/actor_loss" in logger_dict:
            self.actor_losses.append(logger_dict["train/actor_loss"])

        if "train/critic_loss" in logger_dict:
            self.critic_losses.append(logger_dict["train/critic_loss"])

        if "train/entropy_loss" in logger_dict:
            self.entropy_losses.append(logger_dict["train/entropy_loss"])


# =========================
# TRAIN FUNCTION
# =========================
def train_cvar(
    market_df,
    user_df,
    config,
    total_timesteps=300_000,
    model_save_path="outputs/models/cvar_model"
):
    env = DummyVecEnv([lambda: make_env(market_df, user_df, config)])

    # IMPORTANT: stable normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=5.0   # prevents reward spikes
    )

    cvar_callback = CVaRCallback(alpha=0.1)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.999,           # long-term focus
        tau=0.02,
        ent_coef=0.01,         # reduced exploration → safer policy
        verbose=1,
        tensorboard_log="./outputs/tensorboard/cvar/",
        policy_kwargs=dict(net_arch=[128, 128])
    )

    model.learn(total_timesteps=total_timesteps, callback=cvar_callback)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    # =========================
    # LOSS PLOTS
    # =========================
    plt.figure()
    plt.plot(cvar_callback.actor_losses, label="Actor Loss")
    plt.plot(cvar_callback.critic_losses, label="Critic Loss")
    plt.plot(cvar_callback.entropy_losses, label="Entropy Loss")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.title("CVaR-SAC Training Losses (Stable)")
    plt.legend()
    plt.grid()
    plt.show()

    # =========================
    # CVaR PLOT
    # =========================
    if len(cvar_callback.cvar_values) > 10:
        returns = np.array(cvar_callback.cvar_values)

        window = 50
        cvar_series = []

        for i in range(window, len(returns)):
            subset = returns[i - window:i]
            var = np.percentile(subset, 10)
            cvar = subset[subset <= var].mean()
            cvar_series.append(cvar)

        plt.figure()
        plt.plot(cvar_series, label="CVaR (10%)")
        plt.xlabel("Episodes")
        plt.ylabel("CVaR")
        plt.title("Tail Risk (CVaR) During Training")
        plt.legend()
        plt.grid()
        plt.show()

    return model