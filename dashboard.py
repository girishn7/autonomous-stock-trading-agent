import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from rl_agent.trading_env import TradingEnv

st.set_page_config(page_title="Autonomous Trading Agent", layout="wide")
st.title(" Autonomous Stock Trading Agent Dashboard")

# Load data
df = pd.read_csv("data/stock_data_with_regimes.csv")
st.sidebar.markdown("### Data Overview")
st.sidebar.dataframe(df.head())

# Initialize environment and model
env = TradingEnv(df)
model = PPO.load("models/ppo_trading_agent")

# Run simulation
obs = env.reset()
portfolio = []
actions = []

for _ in range(len(df)):
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    portfolio.append(env.balance)
    actions.append(action)
    if done:
        break

# Append actions and portfolio to df for plotting
df = df.iloc[:len(portfolio)]
df['Action'] = actions
df['Portfolio'] = portfolio

# Plot price with actions
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['Close'], label='Close Price')
ax.plot(df['Portfolio'], label='Portfolio Value', linestyle='--')

# Mark buy/sell points
buy_signals = df[df['Action'] == 1]
sell_signals = df[df['Action'] == 2]
ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy', s=50)
ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell', s=50)

ax.set_title("Price, Trades & Portfolio Value")
ax.legend()
st.pyplot(fig)

# Display regime breakdown
st.subheader(" Market Regimes")
st.bar_chart(df['regime'].value_counts())

# Show final portfolio value
st.metric(" Final Portfolio Value", f"${round(portfolio[-1], 2)}")
