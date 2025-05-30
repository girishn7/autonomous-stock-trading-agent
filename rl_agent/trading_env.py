import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        # Observation: OHLCV + regime + position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # 0: none, 1: long
        self.current_step = 0
        self.total_reward = 0
        return self._next_observation()

    def _next_observation(self):
        row = self.data.iloc[self.current_step]
        return np.array([
            row['Open'],
            row['High'],
            row['Low'],
            row['Close'],
            row['Volume'],
            row['regime'],
            self.position
        ], dtype=np.float32)

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        reward = 0

        # Action: 0 = Hold, 1 = Buy, 2 = Sell
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 1:
            reward = current_price - self.entry_price
            self.balance += reward
            self.position = 0
            self.entry_price = 0
            self.total_reward += reward

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._next_observation()

        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")
