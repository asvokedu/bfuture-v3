# ==== utils.py ====
import requests
import pandas as pd
import numpy as np
import ta

def fetch_valid_usdt_symbols():
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()
        symbols = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING' and s['isSpotTradingAllowed']]
        return symbols
    except Exception as e:
        print(f"❌ Gagal fetch simbol dari Binance: {e}")
        return []

def fetch_binance_data(symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "code" in data:
            raise Exception(f"Binance API error: {data.get('msg')}")
    except Exception as e:
        print(f"❌ Error ambil data {symbol} {interval}: {e}")
        return None

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float, errors='ignore')
    return df

def calculate_technical_indicators(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['signal_line'] = macd.macd_signal()
    df['support'] = df['close'].rolling(window=20).min()
    df['resistance'] = df['close'].rolling(window=20).max()
    df.dropna(inplace=True)
    return df

def generate_label(df, reward_threshold=0.0075, risk_threshold=0.004, n_future=3, drop_wait=False, verbose=False):
    df = df.copy()
    labels = []
    close_prices = df["close"].values

    for i in range(len(close_prices)):
        if i + n_future >= len(close_prices):
            labels.append("WAIT")
            continue

        current_price = close_prices[i]
        future_prices = close_prices[i + 1:i + 1 + n_future]

        max_gain = (max(future_prices) - current_price) / current_price
        max_loss = (current_price - min(future_prices)) / current_price

        if max_gain >= reward_threshold:
            labels.append("AGGRESSIVE BUY")
        elif max_loss >= risk_threshold:
            labels.append("SELL")
        else:
            labels.append("WAIT")

    df["label"] = labels

    if verbose:
        label_counts = df["label"].value_counts()
        print(f"\U0001F4D6 Distribusi label: {', '.join([f'{k}: {v}' for k, v in label_counts.items()])}")

    if drop_wait:
        df = df[df["label"] != "WAIT"]

    return df
