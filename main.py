import pandas as pd
from ml_models.clustering import cluster_market_regimes, plot_regimes
from ml_models.classification import train_regime_classifier
from rl_agent.trading_env import TradingEnv
from stable_baselines3 import PPO

# Step 1: Load data
df = pd.read_csv("data/stock_data.csv")

# Step 2: Cluster regimes
df, kmeans_model = cluster_market_regimes(df, n_clusters=3)
plot_regimes(df)

# Step 3: Train classifier
clf = train_regime_classifier(df)

# Step 4: Save enriched data
df.to_csv("data/stock_data_with_regimes.csv", index=False)

# Step 5: Create environment and train PPO agent
env = TradingEnv(df)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("models/ppo_trading_agent")

# Step 6: Simulate trading
obs = env.reset()
for _ in range(len(df)):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
env.render()
