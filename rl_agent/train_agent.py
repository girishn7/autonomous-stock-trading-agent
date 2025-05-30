# rl_agent/train_agent.py
from stable_baselines3 import PPO
from trading_env import TradingEnv
import pandas as pd

data = pd.read_csv("data/stock_data.csv")
env = TradingEnv(data)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("models/ppo_trading_agent")
