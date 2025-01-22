# Swing Trading Bot

This repository contains a swing trading bot that uses historical stock data, technical indicators, and AI to execute trades. Inspired by "Holly" from Trade Ideas, this bot automates swing trading strategies.

## Features

- Fetches historical stock data using Alpaca API.
- Calculates technical indicators (e.g., RSI, MACD).
- Generates buy/sell signals.
- Trains a machine learning model to predict stock movements.
- Executes trades automatically using Alpaca's paper trading API.

## Prerequisites

- Python 3.8+
- An Alpaca account with API keys for paper trading.
- Gitpod or any Python-friendly coding environment.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your_username/swing-trading-bot.git
cd swing-trading-bot
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv trading_env
source trading_env/bin/activate
```

### Step 3: Install Dependencies

Install the required Python packages:

```bash
pip install alpaca-trade-api pandas scikit-learn ta
```

### Step 4: Set Environment Variables

Add your Alpaca API key and secret to the environment:

```bash
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
```

Alternatively, create a `.env` file in the root directory:

```env
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
```

### Step 5: Run the Script

Execute the bot:

```bash
python main.py
```

## Files

- **main.py**: The main script that runs the bot.
- **requirements.txt**: List of required Python packages.
- **README.md**: Documentation for setting up and running the bot.

## Notes

- Ensure your Alpaca account is set to paper trading mode to avoid real trades.
- Adjust the `main.py` parameters to fit your trading strategy.

## License

This project is licensed under the MIT License.

---

For any issues or questions, feel free to open an issue on this repository or contact me directly!