import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment.dynamics.market import MarketDynamics
from environment.dynamics.household import HouseholdDynamics
from environment.dynamics.portfolio import PortfolioDynamics

from environment.state.state_builder import StateBuilder
from environment.reward.reward_function import compute_reward


class FinancialEnv(gym.Env):

    def __init__(self, market_df, user_df, config):
        super().__init__()

        # DATA
        self.market_df = market_df
        self.user_df = user_df
        self.config = config

        # MODULES
        self.market = MarketDynamics(market_df, config)
        self.household = HouseholdDynamics(config)
        self.portfolio = PortfolioDynamics(config)

        self.state_builder = StateBuilder()

        # DIMENSIONS
        self.state_dim = config.STATE_DIM
        self.action_dim = config.ACTION_DIM

        # SPACES
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        # EPISODE CONTROL
        self.current_step = 0

        # STATE
        self.state = None
        self.current_user_idx = None

        # TRACKING VARIABLES
        self.prev_weights = None
        self.initial_wealth = None
        self.expected_income = None

    def _initialize_portfolio_weights(self, user_profile):

        wealth = user_profile["wealth"]
        monthly_expense = user_profile["expense"]
        buffer_target = user_profile["emergency_buffer_months"]  
        risk_appetite = user_profile["risk_appetite"] 

        required_buffer = buffer_target * monthly_expense

        cash_allocation = min(required_buffer / (wealth + 1e-8), 1.0)

        remaining = max(1.0 - cash_allocation, 0.0)

        equity_weight = remaining * risk_appetite
        debt_weight = remaining * (1 - risk_appetite)

        weights = np.array([
            equity_weight,
            debt_weight,
            cash_allocation
        ])

        # Safety normalization
        weights = weights / (weights.sum() + 1e-8)

        return weights


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        # SAMPLE USER
        if hasattr(self, "fixed_user_idx") and self.fixed_user_idx is not None:
            self.current_user_idx = self.fixed_user_idx
        else:
            self.current_user_idx = np.random.randint(len(self.user_df))
        user = self.user_df.iloc[self.current_user_idx]
        self.user_profile = user.to_dict()

        # INITIALIZE MODULES
        market_state = self.market.reset()
        household_state = self.household.reset(user)
        portfolio_state = self.portfolio.reset(household_state)

        # STORE BASELINES
        self.initial_wealth = portfolio_state["wealth"]
        self.expected_income = household_state["income"]

        # INITIAL WEIGHTS
        self.prev_weights = self._initialize_portfolio_weights(self.user_profile)

        # BUILD STATE
        self.state = self.state_builder.build(
            portfolio_state,
            market_state,
            household_state
        )

        self.household_state = household_state
        self.portfolio_state = portfolio_state

        return self._get_obs(), {}

    def step(self, action):

        action = np.clip(action, -1, 1)
        weights = self._normalize_action(action)

        market_state = self.market.step()

        household_state = self.household.step(self.household_state)
        self.household_state = household_state

        portfolio_state = self.portfolio.step(
            prev_state=self.portfolio_state,
            weights=weights,
            market_state=market_state,
            household_state=household_state
        )

        next_state = self.state_builder.build(
            portfolio_state,
            market_state,
            household_state
        )

        reward = compute_reward(
            prev_wealth=self.portfolio_state["wealth"],
            curr_wealth=portfolio_state["wealth"],
            drawdown=portfolio_state["drawdown"],
            buffer_months=portfolio_state["emergency_buffer_months"],
            config=self.config
        )

        self.prev_weights = weights.copy()
        self.portfolio_state = portfolio_state
        self.state = next_state
        self.current_step += 1

        terminated = self._check_termination(next_state)
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, {}

    def _normalize_action(self, action):
        """
        Convert raw action → valid portfolio weights (softmax)
        """
        action = np.exp(action - np.max(action))
        weights = action / (np.sum(action) + 1e-8)
        return weights.astype(np.float32)

    def _get_obs(self):
        return self.state_builder.to_vector(self.state).astype(np.float32)

    def _check_termination(self, state):
        wealth = state["wealth"]
        goal = state["goal"]

        if wealth <= 1.0:
            return True

        if goal <= 0:
            return True

        return False