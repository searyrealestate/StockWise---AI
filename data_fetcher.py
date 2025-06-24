# data_fetcher.py

import requests
import pandas as pd

ALPHA_API_KEY = "A7HSI5P9ZA7PKF3D"
FINNHUB_API_KEY = "d1d7ochr01qic6lhn8ggd1d7ochr01qic6lhn8h0"

def get_stock_history(symbol, interval="Daily", outputsize="full"):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": ALPHA_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        raise ValueError("הנתונים לא התקבלו כראוי:", data)

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
    df.sort_index(inplace=True)
    df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. adjusted close": "Adj Close",
        "6. volume": "Volume"
    }, inplace=True)
    return df

def get_company_news(symbol, from_date, to_date):
    url = f"https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY
    }
    res = requests.get(url, params=params)
    return res.json()
