from dotenv import load_dotenv
import os
import time
import pandas as pd
import ta
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # For paper trading
SYMBOL = "AAPL"
TIMEFRAME = TimeFrame.Day
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
LOOKBACK_WINDOW = 20  # For rolling window calculations
TEST_SIZE = 0.2  # For model evaluation

# Initialize Alpaca API
api = REST(API_KEY, SECRET_KEY, base_url=BASE_URL)

# Step 1: Fetch Historical Data
def fetch_data(symbol, start_date, end_date, timeframe):
    bars = api.get_bars(symbol, timeframe, start_date, end_date).df
    return bars

# Step 2: Generate Technical Indicators
def add_indicators(data):
    data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
    data['sma'] = data['close'].rolling(window=50).mean()
    data['macd'], _, _ = ta.trend.macd(data['close'])
    return data

# Step 3: Generate Trading Signals
def generate_signals(data):
    data['signal'] = 0
    data.loc[(data['close'] > data['sma']) & (data['rsi'] < 70), 'signal'] = 1  # Buy
    data.loc[(data['close'] < data['sma']) & (data['rsi'] > 30), 'signal'] = -1  # Sell
    return data

# Step 4: Prepare Data for Model Training
def prepare_data(data):
    data['target'] = data['close'].shift(-1) > data['close']
    features = ['rsi', 'sma', 'macd']
    X = data[features].dropna()
    y = data['target'].dropna().shift(-1)
    return X, y

# Step 5: Train AI Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    return model

# Step 6: Predict Trends
def predict_trend(data, model):
    features = data[['rsi', 'sma', 'macd']]
    data['prediction'] = model.predict(features)
    return data

# Step 7: Execute Trades (Simulated)
def execute_trade(signal, position):
    if signal == 1 and position == 0:
        print("Buy Order Executed")
    elif signal == -1 and position == 1:
        print("Sell Order Executed")
    return signal

# Main Function
def main():
    print("Fetching data...")
    bars = fetch_data(SYMBOL, START_DATE, END_DATE, TIMEFRAME)
    print("Adding indicators...")
    bars = add_indicators(bars)
    print("Generating signals...")
    bars = generate_signals(bars)

    print("Preparing data for model training...")
    X, y = prepare_data(bars)
    print("Training model...")
    model = train_model(X, y)

    # Backtesting Loop
    position = 0  # 0: No position, 1: Long
    for index, row in bars.iloc[LOOKBACK_WINDOW:].iterrows():
        signal = row['prediction']
        position = execute_trade(signal, position)

if __name__ == "__main__":
    main()