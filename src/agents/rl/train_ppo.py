import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from environment.financial_env import FinancialEnv


def make_env(market_df, user_df, config):
    return FinancialEnv(market_df, user_df, config)


class LossTrackingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        logger_dict = self.model.logger.name_to_value

        if "train/policy_gradient_loss" in logger_dict:
            self.policy_losses.append(logger_dict["train/policy_gradient_loss"])

        if "train/value_loss" in logger_dict:
            self.value_losses.append(logger_dict["train/value_loss"])

        if "train/entropy_loss" in logger_dict:
            self.entropy_losses.append(logger_dict["train/entropy_loss"])


def train_ppo(
    market_df,
    user_df,
    config,
    total_timesteps=300_000,
    model_save_path="outputs/models/ppo_model"
):
    env = DummyVecEnv([lambda: make_env(market_df, user_df, config)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 🔹 Initialize callback
    loss_callback = LossTrackingCallback()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log="./outputs/tensorboard/ppo/",
        policy_kwargs=dict(net_arch=[128, 128])
    )

    model.learn(total_timesteps=total_timesteps, callback=loss_callback)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    plt.figure()
    plt.plot(loss_callback.policy_losses, label="Policy Loss")
    plt.plot(loss_callback.value_losses, label="Value Loss")
    plt.plot(loss_callback.entropy_losses, label="Entropy Loss")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.title("PPO Training Losses")
    plt.legend()
    plt.grid()
    plt.show()

    return model