# ==== evaluate_performance.py ====
def evaluate_signal_roi(df, signal_type="AGGRESSIVE BUY", lookahead=6):
    current_price = df["close"].iloc[-1]
    future_prices = df["close"].iloc[-lookahead:]
    max_price = future_prices.max()
    min_price = future_prices.min()

    if signal_type == "AGGRESSIVE BUY":
        roi = (max_price - current_price) / current_price
    elif signal_type == "SELL":
        roi = (current_price - min_price) / current_price
    else:
        roi = 0.0

    return {"roi": roi, "max_price": max_price, "min_price": min_price}

def log_backtest_performance(symbol, interval, signal_type, roi_info):
    print(f"📈 Evaluasi ROI {symbol} @ {interval} untuk sinyal {signal_type}:")
    print(f"  ROI: {roi_info['roi']:.2%}, High: {roi_info['max_price']}, Low: {roi_info['min_price']}")
