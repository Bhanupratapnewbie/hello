from flask import Flask, render_template_string
from track_accounts import load_accounts, track_accounts

TEMPLATE = """
<!doctype html>
<title>TradingView Dashboard</title>
<h1>TradingView Account Tracker</h1>
{% for account, tickers in data.items() %}
<h2>{{ account }}</h2>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Ticker</th><th>Recommendation</th><th>Time</th><th>Error</th></tr>
{% for tick in tickers %}
<tr>
  <td>{{ tick['symbol'] }}</td>
  <td>{{ tick.get('summary', {}).get('RECOMMENDATION', 'N/A') }}</td>
  <td>{{ tick.get('time', '') }}</td>
  <td>{{ tick.get('error', '') }}</td>
</tr>
{% endfor %}
</table>
{% endfor %}
"""

app = Flask(__name__)

@app.route('/')
def index():
    accounts = load_accounts()
    data = track_accounts(accounts)
    return render_template_string(TEMPLATE, data=data)

if __name__ == '__main__':
    app.run(debug=True)
