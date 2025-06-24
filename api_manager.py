# api_manager.py

from datetime import date
from collections import defaultdict

API_KEYS = {
    "alphavantage": "A7HSI5P9ZA7PKF3D",
    "finnhub": "d1d7ochr01qic6lhn8ggd1d7ochr01qic6lhn8h0",
    "polygon": "ZQ9rwnJb0K3tc1YkIPv1qUgOvACtdM6g"
}

DAILY_LIMITS = {
    "alphavantage": 500,   # לפי מסלול חינמי
    "finnhub": 60,
    "polygon": 1000
}

class APIManager:
    def __init__(self):
        self.usage = defaultdict(int)
        self.last_reset = date.today()

    def _reset_if_needed(self):
        if self.last_reset != date.today():
            self.usage = defaultdict(int)
            self.last_reset = date.today()

    def get_available_api(self):
        self._reset_if_needed()
        for api_name, key in API_KEYS.items():
            if self.usage[api_name] < DAILY_LIMITS[api_name]:
                self.usage[api_name] += 1
                return api_name, key
        return None, None

    def get_usage_summary(self):
        self._reset_if_needed()
        return dict(self.usage)
