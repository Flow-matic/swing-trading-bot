# swing_trading_bot/main.py

# Required Libraries
import pandas as pd
import ta
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.ensemble import RandomForestClassifier
import time
import os

# Configuration
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'
SYMBOL = "AAPL"
TIMEFRAME = TimeFrame.Day
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

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
    return data

# Step 3: Generate Trading Signals
def generate_signals(data):
    data['signal'] = 0
    data.loc[data['close'] > data['sma'], 'signal'] = 1  # Buy
    data.loc[data['close'] < data['sma'], 'signal'] = -1  # Sell
    return data

# Step 4: Train AI Model
def train_model(data):
    features = data[['rsi', 'sma']].dropna()
    target = (data['close'].shift(-1) > data['close']).astype(int)
    model = RandomForestClassifier()
    model.fit(features[:-1], target[:-1])
    return model

# Step 5: Predict Trends
def predict_trend(data, model):
    features = data[['rsi', 'sma']].dropna()
    data['prediction'] = model.predict(features)
    return data

# Step 6: Execute Trades
def execute_trade(signal):
    if signal == 1:
        api.submit_order(symbol=SYMBOL, qty=1, side="buy", type="market", time_in_force="gtc")
        print("Buy Order Executed")
    elif signal == -1:
        api.submit_order(symbol=SYMBOL, qty=1, side="sell", type="market", time_in_force="gtc")
        print("Sell Order Executed")

# Main Function
def main():
    print("Fetching data...")
    bars = fetch_data(SYMBOL, START_DATE, END_DATE, TIMEFRAME)
    print("Adding indicators...")
    bars = add_indicators(bars)
    print("Generating signals...")
    bars = generate_signals(bars)
    print("Training model...")
    model = train_model(bars)
    print("Predicting trends...")
    bars = predict_trend(bars, model)

    # Live Trading Loop
    while True:
        print("Fetching latest data...")
        latest_bars = fetch_data(SYMBOL, "2023-12-01", "2023-12-31", TIMEFRAME)  # Adjust dates
        latest_bars = add_indicators(latest_bars)
        latest_signal = latest_bars.iloc[-1]['signal']

        print(f"Latest Signal: {latest_signal}")
        execute_trade(latest_signal)

        time.sleep(3600)  # Wait for an hour before the next check

if __name__ == "__main__":
    main()