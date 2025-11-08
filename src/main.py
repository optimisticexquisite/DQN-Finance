import numpy as np
from dqn_finance import MarketEnvironment, create_agent
import pandas as pd
# data: numpy array or DataFrame with OHLCV columns (close price at index/name expected)
data = pd.read_csv("/home/rahuladhikar/DQN-Finance/src/data/Nifty_50.csv", usecols = ["Price","Open","High","Low","Vol."])
# === PATCH for NIFTY50 CSV format ===
data.columns = [col.strip().replace('.', '').replace('%', '').lower() for col in data.columns]
rename_map = {
    "price": "close",
    "vol": "volume",
    "date": "timestamp"
}
data = data.rename(columns=rename_map)

# Clean numeric columns
for col in ["open", "high", "low", "close", "volume"]:
    if col in data.columns:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("M", "e6", regex=False)
            .str.replace("%", "", regex=False)
        )
        data[col] = pd.to_numeric(data[col], errors="coerce")

# --- FIX PART 2: HANDLE THE np.nan VALUES ---
# Now, we deal with the np.nan values that "coerce" found.

# 1. Forward-fill: Assumes a missing value is the same as the last known one.
data = data.fillna(method="ffill")

# 2. Drop remaining: If "nan" was at the very start, ffill can't fix it.
#    This drops those initial bad rows.
data = data.dropna()

# 3. Good practice after dropping rows
data = data.reset_index(drop=True)
# --- END FIX ---

if "timestamp" in data.columns:
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%d-%m-%Y", errors="coerce")

env = MarketEnvironment(data, lookback = 60, stabilization_window = 5)
agent = create_agent(env, "H1")  # automatically uses preset hyperparameters
history = agent.fit(env, epochs = 100)
for i in range(len(history)):
    print(history[i])