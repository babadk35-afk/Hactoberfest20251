"""
Time Series Forecast (ARIMA)
- Loads a CSV with 'date,value' columns, fits ARIMA, plots forecast vs history
Usage:
  python 18_arima_forecast.py data.csv 30
Dependencies: pandas, matplotlib, statsmodels
"""
import sys, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

if __name__ == "__main__":
    if len(sys.argv)<3: print(__doc__); sys.exit(0)
    path, horizon = sys.argv[1], int(sys.argv[2])
    df = pd.read_csv(path, parse_dates=['date']).sort_values('date')
    model = ARIMA(df['value'], order=(2,1,2)).fit()
    f = model.forecast(steps=horizon)
    plt.figure(); plt.plot(df['date'], df['value'], label="history")
    future_idx = range(len(df), len(df)+horizon)
    plt.plot(future_idx, f, label="forecast")
    plt.title("ARIMA Forecast"); plt.legend(); plt.show()
