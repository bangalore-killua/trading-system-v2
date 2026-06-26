import logging
logging.basicConfig(level=logging.WARNING)

from strategy.training_pipeline import AutoGluonPipeline
from backtester.intelligent_backtest import run_backtest

pipe = AutoGluonPipeline(symbols=["AAPL","NVDA","MSFT"], years=2)
all_records = []
for sym in ["AAPL","NVDA","MSFT"]:
    r = run_backtest(symbol=sym, years=2)
    all_records.extend(r.get("records", []))

df = pipe._records_to_features(all_records, model_type="dip")
counts = df["label"].value_counts().to_dict()
print("Total dip records (rsi_14 < 40):", len(df))
print("Label counts:", counts)
print("Both classes present:", set(df["label"].unique()) == {0, 1})
if len(df) >= 100 and set(df["label"].unique()) == {0, 1}:
    print("PASS - safe to launch dip model training")
else:
    print("FAIL - check data")
