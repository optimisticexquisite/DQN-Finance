import numpy as np
from dqn_finance import MarketEnvironment, create_agent
import pandas as pd
# data: numpy array or DataFrame with OHLCV columns (close price at index/name expected)
data = pd.read_csv("/home/rahuladhikar/DQN-Finance/src/data/mock_ohlcv.csv")
env = MarketEnvironment(data, lookback=60, stabilization_window=5)
agent = create_agent(env, "H1")  # automatically uses preset hyperparameters
history = agent.fit(env, epochs=100)
for i in range(len(history)):
    print(history[i])