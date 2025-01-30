import time
import requests
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Stock API endpoint (Alpha Vantage for stocks, removed crypto)
URL = "https://www.alphavantage.co/query"
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Initialize data
price_history = []
volume_history = []
features = []
labels = []
model = None
scaler = None  # Initialize scaler to None

# Logging configuration
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model():
    global model, scaler
    if os.path.exists("stock_model.h5") and os.path.exists("scaler.pkl"):
        model = create_model()
        model.load_weights("stock_model.h5")
        scaler = joblib.load("scaler.pkl")
        print("Model and scaler loaded from file.")
    else:
        print("No saved model found. Training required.")

def save_data():
    np.save("price_history.npy", price_history)
    np.save("volume_history.npy", volume_history)
    np.save("features.npy", features)
    np.save("labels.npy", labels)
    print("Historical data saved.")

def load_data():
    global price_history, volume_history, features, labels
    if os.path.exists("price_history.npy"):
        price_history = np.load("price_history.npy").tolist()
    if os.path.exists("volume_history.npy"):
        volume_history = np.load("volume_history.npy").tolist()
    if os.path.exists("features.npy"):
        features = np.load("features.npy").tolist()
    if os.path.exists("labels.npy"):
        labels = np.load("labels.npy").tolist()
    print("Historical data loaded.")

def get_stock_data(symbol):
    try:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "5min",
            "apikey": API_KEY
        }
        response = requests.get(URL, params=params)
        response.raise_for_status()
        data = response.json()
        latest_time = list(data["Time Series (5min)"].keys())[0]
        latest_data = data["Time Series (5min)"][latest_time]
        return {
            "last_price": float(latest_data["4. close"]),
            "volume": float(latest_data["5. volume"]),
            "high": float(latest_data["2. high"]),
            "low": float(latest_data["3. low"]),
            "close": float(latest_data["4. close"])
        }
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        print(f"Error fetching stock data: {e}")
        return None

def execute_bot(symbol):
    global model, scaler
    market_data = get_stock_data(symbol)
    if market_data:
        price_history.append(market_data["last_price"])
        volume_history.append(market_data["volume"])

        print(f"Price history length: {len(price_history)}, Volume history length: {len(volume_history)}")

        if len(price_history) >= 21:
            features_row = calculate_technical_indicators()
            if features_row:
                label = 1 if price_history[-1] > price_history[-2] else 0
                features.append(list(features_row.values()))
                labels.append(label)
                save_data()

                if model is None and len(features) >= 50:
                    train_model()

                if model and scaler:
                    try:
                        features_scaled = scaler.transform([list(features_row.values())]).reshape(1, 1, -1)
                        prediction = model.predict(features_scaled)[0][0]

                        direction = "UP" if prediction > 0.5 else "DOWN"
                        confidence = abs(prediction - 0.5) * 2

                        now = datetime.now()
                        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

                        print("=" * 50)
                        print(f"[{timestamp}] Stock Price Prediction")
                        print("=" * 50)
                        print(f"Current Price: {market_data['last_price']:.2f}")
                        print(f"Prediction: Price will go {direction} in the next epoch.")
                        print(f"Confidence: {confidence * 100:.2f}%")
                        print("=" * 50)
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                        logging.exception("Error during prediction")

if __name__ == "__main__":
    load_data()
    load_model()
    print(f"[{datetime.now()}] Bot started.")
    while True:
        try:
            execute_bot("AAPL")  # Example: Predict Apple stock (AAPL)
            time.sleep(300)  # 5-minute interval to match stock market data update
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"Error: {e}")