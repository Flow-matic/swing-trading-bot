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

load_dotenv()

# API and data
ALPHA_VANTAGE_API_KEY = ""
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if ALPHA_VANTAGE_API_KEY is None:
    raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set!")

STOCK_SYMBOL = "AAPL"
INTRADAY_INTERVAL = "5min"  # Make this configurable
PRICE_HISTORY_FILE = "price_history.npy"
VOLUME_HISTORY_FILE = "volume_history.npy"
FEATURES_FILE = "features.npy"
LABELS_FILE = "labels.npy"
MODEL_FILE = "stock_model.h5"
SCALER_FILE = "scaler.pkl"

# Initialize
price_history = []
volume_history = []
features = []
labels = []
model = None
scaler = None

# Logging
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- API Handling and Rate Limiting ---
def get_market_data(retries=3, retry_delay=60):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={STOCK_SYMBOL}&interval={INTRADAY_INTERVAL}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}&datatype=json"

    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if "Error Message" in data:
                raise ValueError(data["Error Message"])

            if "Information" in data and "Thank you for using Alpha Vantage" in data["Information"]:  # Rate limited
                wait_time = retry_delay * (2**attempt) # Exponential backoff
                print(f"Rate limited. Retrying in {wait_time} seconds (attempt {attempt + 1}/{retries})...")
                time.sleep(wait_time)
                continue # next retry attempt

            return data  # Success!

        except requests.exceptions.RequestException as e:
            print(f"Request Error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                raise  # Re-raise the exception after all retries fail
        except json.JSONDecodeError as e:
            print(f"JSON Error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    return None  # All retries failed

# --- Data Loading/Saving ---
def load_model():
    global model, scaler
    try:
        model = Sequential() # You must initialize the model architecture before loading weights.
        model.add(LSTM(50, activation='relu', input_shape=(1, 10))) # Example: Assuming 10 features
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse') # You must compile the model before loading weights.

        model.load_weights(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        print("Model and scaler loaded.")
    except (OSError, ValueError) as e:  # Handle file not found or other errors during loading
        print(f"Error loading model: {e}. Training required.")

def save_model():
    if model and scaler:
        model.save_weights(MODEL_FILE)  # Save only weights
        joblib.dump(scaler, SCALER_FILE)
        print("Model and scaler saved.")

def load_data():
    global price_history, volume_history, features, labels
    try:
        price_history = np.load(PRICE_HISTORY_FILE).tolist()
        volume_history = np.load(VOLUME_HISTORY_FILE).tolist()
        features = np.load(FEATURES_FILE).tolist()
        labels = np.load(LABELS_FILE).tolist()
        print("Historical data loaded.")
    except OSError:
        print("No historical data found. Starting from scratch.")

def save_data():
    np.save(PRICE_HISTORY_FILE, price_history)
    np.save(VOLUME_HISTORY_FILE, volume_history)
    np.save(FEATURES_FILE, features)
    np.save(LABELS_FILE, labels)
    print("Historical data saved.")

# --- Technical Indicators ---
def calculate_technical_indicators():
    df = pd.DataFrame({
        "close": price_history[-20:],  # Use a rolling window (e.g., 20 periods)
        "volume": volume_history[-20:]
    })

    # Example: Simple Moving Average (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()

    # Example: Relative Strength Index (RSI)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    df['RSI_14'] = up.rolling(window=14).mean() / down.rolling(window=14).mean() * 100

    # Example: Volume Average
    df['Volume_Average_5'] = df['volume'].rolling(window=5).mean()

    # ... Add more indicators as needed ...

    return df.iloc[-1].to_dict()  # Return the last row as a dictionary

# --- Model Training ---
def create_model(): # This function is crucial and was missing.
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 10))) # Example: Assuming 10 features
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model():
    global model, scaler
    features_np = np.array(features)
    labels_np = np.array(labels)

    # Check for minimum data points for training
    if len(features_np) < 50:
        print("Not enough data points for training.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features_np, labels_np, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for LSTM (samples, timesteps, features)
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    model = create_model() # Create the model instance
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32) # Adjust epochs and batch size

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    save_model()

# --- Bot Execution ---
def execute_bot():
    global model, scaler