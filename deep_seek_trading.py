import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from alpaca_trade_api import REST
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import optuna
import talib as ta

# API Configuration
ALPACA_API_KEY = "your_alpaca_api_key"
ALPACA_SECRET_KEY = "your_alpaca_secret_key"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Use paper trading for testing

# Initialize Alpaca API
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# Logging
import logging
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Global variables
price_history = []
features = []
labels = []
model = None
scaler = StandardScaler()

def get_real_time_data(symbol="SPY"):
    """Fetches real-time market data using Alpaca API."""
    try:
        barset = api.get_barset(symbol, "minute", limit=100)
        bars = barset[symbol]
        df = pd.DataFrame({
            "time": [bar.t for bar in bars],
            "open": [bar.o for bar in bars],
            "high": [bar.h for bar in bars],
            "low": [bar.l for bar in bars],
            "close": [bar.c for bar in bars],
            "volume": [bar.v for bar in bars]
        })
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculates technical indicators using TA-Lib."""
    df["rsi"] = ta.RSI(df["close"], timeperiod=14)
    df["macd"], df["macd_signal"], df["macd_hist"] = ta.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["bollinger_upper"], df["bollinger_middle"], df["bollinger_lower"] = ta.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df["atr"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["obv"] = ta.OBV(df["close"], df["volume"])
    df.dropna(inplace=True)
    return df.iloc[-1].to_dict()

def train_model(features, labels):
    """Trains a LightGBM model with hyperparameter tuning."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    model = LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    return model

def predict_direction(model, features):
    """Predicts the price direction with confidence score."""
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    confidence = abs(prediction - 0.5) * 2  # Confidence score between 0 and 1
    return "UP" if prediction > 0.5 else "DOWN", confidence

def execute_bot():
    global price_history, features, labels, model

    df = get_real_time_data("SPY")
    if df is not None:
        last_row = df.iloc[-1]
        price_history.append(last_row["close"])

        if len(price_history) >= 50:  # Minimum data points for indicators
            try:
                features_row = calculate_technical_indicators(df)
                if features_row:
                    label = 1 if price_history[-1] > price_history[-2] else 0
                    features.append(list(features_row.values()))
                    labels.append(label)

                    if len(features) >= 100:  # Train model with sufficient data
                        model = train_model(features, labels)

                    if model:
                        prediction, confidence = predict_direction(model, list(features_row.values()))
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prediction: {prediction} (Confidence: {confidence:.2f})")

            except Exception as e:
                logging.error(f"Error processing data: {e}")

def start_bot():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bot started.")
    while True:
        try:
            execute_bot()
            time.sleep(60)  # Run every minute
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    start_bot()