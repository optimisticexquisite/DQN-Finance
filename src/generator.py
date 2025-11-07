from dqn_finance import GBMGARCHParams, generate_mock_ohlcv, GBMParams

# params = GBMGARCHParams(drift=0.08, omega=1e-6, alpha=0.04, beta=0.9, initial_volatility=0.015)
params = GBMParams(drift=0.07, volatility=0.25)
df = generate_mock_ohlcv(
    5000,
    start_price = 120.0,
    params = params,
    freq = "15T",
    base_volume = 5e5,
    seed = 42,
)
df.to_csv("data/mock_ohlcv_1.csv", index=False)
