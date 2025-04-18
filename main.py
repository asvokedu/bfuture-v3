# === main.py ===
import os
import time
import joblib
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from train_model import train_model_for_symbol
from utils import fetch_binance_data, calculate_technical_indicators, fetch_symbols
from evaluate_performance import evaluate_signal_roi, log_backtest_performance

MODEL_DIR = "models"
INTERVALS = ["1h", "4h"]
MIN_ROWS_TO_PREDICT = 50

def wait_until_next_candle(interval):
    now = datetime.utcnow()
    if interval.endswith("h"):
        hours = int(interval[:-1])
        next_candle = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours))
    elif interval.endswith("m"):
        minutes = int(interval[:-1])
        next_candle = (now - timedelta(minutes=now.minute % minutes,
                                       seconds=now.second,
                                       microseconds=now.microsecond)
                       + timedelta(minutes=minutes))
    else:
        print(f"❌ Interval {interval} tidak dikenali untuk sinkronisasi waktu.")
        return

    wait_seconds = (next_candle - now).total_seconds()
    while wait_seconds > 0:
        mins, secs = divmod(int(wait_seconds), 60)
        print(f"\r⏳ Menunggu candle baru untuk interval {interval}: {mins:02d}:{secs:02d} ", end="", flush=True)
        time.sleep(1)
        wait_seconds -= 1
    print(f"\n✅ Mulai analisa pada awal candle {interval}")

def analyze_symbol(symbol):
    for interval in INTERVALS:
        print(f"\n🔍 Menganalisis {symbol} @ {interval}...")

        df = fetch_binance_data(symbol, interval)
        if df is None or len(df) < MIN_ROWS_TO_PREDICT:
            print(f"⚠️ Data {interval} untuk {symbol} terlalu sedikit.")
            continue

        df = calculate_technical_indicators(df)
        latest = df.iloc[-1:]

        model_filename = f"{symbol}_{interval}.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)

        if not os.path.exists(model_path):
            print(f"⚠️ Model belum tersedia untuk {symbol}, melatih baru...")
            train_model_for_symbol(symbol, interval)

        if not os.path.exists(model_path):
            print(f"❌ Gagal memuat model/encoder untuk {symbol} setelah pelatihan.")
            print(f"🔮 Prediksi fallback: WAIT")
            continue

        try:
            model = joblib.load(model_path)
            X_pred = latest[["rsi", "macd", "signal_line", "support", "resistance", "volume"]]
            pred = model.predict(X_pred)[0]

            label_map = {0: "AGGRESSIVE BUY", 1: "SELL", 2: "WAIT"} if hasattr(model, "classes_") and len(model.classes_) == 3 else {0: "SELL", 1: "AGGRESSIVE BUY"}
            pred_label = label_map.get(pred, "UNKNOWN")

            print(f"✅ Prediksi {symbol} ({interval}): {pred_label}")

            if pred_label in ["AGGRESSIVE BUY", "SELL"]:
                roi_info = evaluate_signal_roi(df, signal_type=pred_label)
                log_backtest_performance(symbol, interval, pred_label, roi_info)

        except Exception as e:
            print(f"❌ Gagal analisa {symbol} @ {interval}: {e}")
            print(f"🔮 Prediksi fallback: WAIT")

def main_loop():
    while True:
        print(f"\n🕒 Sinkronisasi waktu pada {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

        # Tunggu hingga waktu candle baru untuk interval 1h (utama)
        wait_until_next_candle("1h")

        print(f"\n🚀 Mulai analisa jam {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        symbols = fetch_symbols()

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(analyze_symbol, symbols)

        print("✅ Analisa selesai.\n")

if __name__ == "__main__":
    main_loop()
