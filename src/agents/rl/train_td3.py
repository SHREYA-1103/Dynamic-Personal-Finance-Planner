import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from environment.financial_env import FinancialEnv


def make_env(market_df, user_df, config):
    return FinancialEnv(market_df, user_df, config)


class TD3LossTrackingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.actor_losses = []
        self.critic_losses = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        logger_dict = self.model.logger.name_to_value

        if "train/actor_loss" in logger_dict:
            self.actor_losses.append(logger_dict["train/actor_loss"])

        if "train/critic_loss" in logger_dict:
            self.critic_losses.append(logger_dict["train/critic_loss"])


def train_td3(
    market_df,
    user_df,
    config,
    total_timesteps=300_000,
    model_save_path="outputs/models/td3_model"
):
    env = DummyVecEnv([lambda: make_env(market_df, user_df, config)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    loss_callback = TD3LossTrackingCallback()

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.995,
        tau=0.005,
        policy_delay=2,
        target_policy_noise=0.1,
        target_noise_clip=0.2,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        tensorboard_log="./outputs/tensorboard/td3/"
    )

    model.learn(total_timesteps=total_timesteps, callback=loss_callback)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    plt.figure()
    plt.plot(loss_callback.actor_losses, label="Actor Loss")
    plt.plot(loss_callback.critic_losses, label="Critic Loss")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.title("TD3 Training Losses")
    plt.legend()
    plt.grid()
    plt.show()

    return model