# TradingView Account Tracker

This small Python script helps track data from multiple TradingView accounts by listing the tickers you follow in each account.

## Setup

1. Install dependencies:
   ```bash
   pip install tradingview_ta flask
   ```
2. Edit `accounts.json` and list each account's name and tickers.

## Usage

Run the tracker:

```bash
python track_accounts.py
```

The script prints a JSON summary of the technical analysis for each ticker.

### Run the Dashboard

If you prefer a simple web dashboard, run:

```bash
python dashboard.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser to view
the account summaries.
