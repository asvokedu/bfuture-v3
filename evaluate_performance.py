# ===== evaluate_performance.py =====
def evaluate_signal_roi(df, signal_type, lookahead=6):
    try:
        entry_price = df.iloc[-1]["close"]
        future = df["close"].iloc[-lookahead:]
        if future.empty:
            return {"roi": 0.0}

        if signal_type == "AGGRESSIVE BUY":
            max_future_price = future.max()
            roi = (max_future_price - entry_price) / entry_price
        elif signal_type == "SELL":
            min_future_price = future.min()
            roi = (entry_price - min_future_price) / entry_price
        else:
            roi = 0.0
        return {"roi": roi}
    except Exception as e:
        print(f"❌ Error menghitung ROI: {e}")
        return {"roi": 0.0}

def log_backtest_performance(symbol, interval, label, roi_info):
    roi = roi_info.get("roi", 0)
    print(f"📈 ROI simulasi {symbol} ({interval}) untuk {label}: {roi*100:.2f}%")
