import numpy as np
from dqn_finance import MarketEnvironment, create_agent

# data: numpy array or DataFrame with OHLCV columns (close price at index/name expected)
env = MarketEnvironment(data, lookback=60, stabilization_window=5)
agent = create_agent(env, "H1")  # automatically uses preset hyperparameters

history = agent.fit(env, epochs=10)
print(history[-1])