# api_manager.py

from datetime import date
from collections import defaultdict
import nasdaqdatalink
import os
from dotenv import load_dotenv

load_dotenv()


class APIManager:
    def __init__(self):
        self.usage = defaultdict(int)
        self.last_reset = date.today()

        self.API_KEYS = {
            "alphavantage": os.getenv("ALPHAVANTAGE_KEY"),
            "finnhub": os.getenv("FINNHUB_KEY"),
            "polygon": os.getenv("POLYGON_KEY"),
            "sheetson": os.getenv("SHEETSON_KEY"),
            "API_Nasdaq_data_link": os.getenv("NASDAQ_DATA_LINK_KEY")
        }

        self.DAILY_LIMITS = {
            "alphavantage": 500,  # לפי מסלול חינמי
            "finnhub": 60,
            "polygon": 1000,
            "sheetson": 1000,
            "API_Nasdaq_data_link": 10000  # or whatever your plan allows

        }

    def _reset_if_needed(self):
        if self.last_reset != date.today():
            self.usage = defaultdict(int)
            self.last_reset = date.today()

    def get_usage_summary(self):
        self._reset_if_needed()
        return dict(self.usage)

    def get_available_api(self):
        self._reset_if_needed()
        for api_name, key in self.API_KEYS.items():
            if self.usage[api_name] < self.DAILY_LIMITS.get(api_name, float('inf')):
                self.usage[api_name] += 1
                print(f"🔄 Using {api_name} (used {self.usage[api_name]}/{self.DAILY_LIMITS[api_name]})")
                return api_name, key
        return None, None

    def get_nasdaq_key(self):
        self._reset_if_needed()
        key = self.API_KEYS.get("API_Nasdaq_data_link")
        if self.usage["API_Nasdaq_data_link"] < self.DAILY_LIMITS.get("API_Nasdaq_data_link", float('inf')):
            self.usage["API_Nasdaq_data_link"] += 1
            return key
        return None

    def get_key(self, name):
        return self.API_KEYS.get(name)


if __name__ == "__main__":
    manager = APIManager()
    nasdaqdatalink.ApiConfig.api_key = manager.get_nasdaq_key()
