# ===== train_model.py =====
import os
import joblib
import optuna
import warnings
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from utils import fetch_binance_data, calculate_technical_indicators, generate_label

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


def train_model_for_symbol(symbol, interval):
    df = fetch_binance_data(symbol, interval)
    if df is None or len(df) < 60:
        print(f"❌ Data untuk {symbol} terlalu sedikit untuk pelatihan.")
        return

    df = calculate_technical_indicators(df)
    df = generate_label(df, verbose=True)
    if df.empty or len(df.label.unique()) < 2:
        print(f"❌ Label tidak memadai untuk {symbol}, pelatihan dibatalkan.")
        return

    X = df[["rsi", "macd", "signal_line", "support", "resistance", "volume"]]
    y = df["label"]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric="mlogloss")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average="weighted")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    best_params = study.best_trial.params
    best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="mlogloss")
    best_model.fit(X, y_encoded)

    model_path = os.path.join(MODEL_DIR, f"{symbol}_{interval}.pkl")
    joblib.dump(best_model, model_path)
    print(f"💾 Model disimpan: {model_path}")
