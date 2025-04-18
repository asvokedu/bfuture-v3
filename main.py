# ==== main.py ====
import os
import joblib
from train_model import train_model_for_symbol
from utils import fetch_binance_data, calculate_technical_indicators, fetch_valid_usdt_symbols
from evaluate_performance import evaluate_signal_roi, log_backtest_performance

MODEL_DIR = "models"
INTERVALS = ["1h", "4h"]
MIN_ROWS_TO_PREDICT = 50

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

def main():
    symbols = fetch_valid_usdt_symbols()
    print(f"🔎 Memproses {len(symbols)} simbol valid dari Binance")
    for symbol in symbols:
        analyze_symbol(symbol)

if __name__ == "__main__":
    main()
