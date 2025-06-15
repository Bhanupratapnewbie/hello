import json
from tradingview_ta import TA_Handler, Interval
from datetime import datetime
from pathlib import Path

CONFIG_FILE = 'accounts.json'


def load_accounts(config_file=CONFIG_FILE):
    with open(config_file, 'r') as f:
        return json.load(f)


def get_ticker_summary(symbol):
    handler = TA_Handler(symbol=symbol, screener='crypto', exchange='BINANCE', interval=Interval.INTERVAL_1_DAY)
    try:
        analysis = handler.get_analysis()
        return {
            'symbol': symbol,
            'time': datetime.utcnow().isoformat(),
            'summary': analysis.summary
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'error': str(e)
        }


def track_accounts(accounts):
    results = {}
    for account in accounts:
        name = account.get('name')
        tickers = account.get('tickers', [])
        results[name] = [get_ticker_summary(t) for t in tickers]
    return results


def main():
    accounts = load_accounts()
    data = track_accounts(accounts)
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
