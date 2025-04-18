# ==== train_model.py ====
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from utils import fetch_binance_data, calculate_technical_indicators, generate_label

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model_for_symbol(symbol, interval):
    df = fetch_binance_data(symbol, interval, limit=150)
    if df is None or len(df) < 50:
        print(f"❌ Data untuk {symbol} terlalu sedikit untuk pelatihan.")
        return

    df = calculate_technical_indicators(df)
    df = generate_label(df, drop_wait=True)

    if df.empty:
        print(f"⚠️ Tidak cukup data berlabel untuk {symbol}.")
        return

    features = ["rsi", "macd", "signal_line", "support", "resistance", "volume"]
    X = df[features]
    y = df["label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_enc)

    model_filename = f"{symbol}_{interval}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(model, model_path)

    print(f"✅ Model untuk {symbol} ({interval}) disimpan ke {model_path}")
