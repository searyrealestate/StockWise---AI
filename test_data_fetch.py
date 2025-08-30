from data_fetcher import get_stock_history, get_company_news
from pprint import pprint

df = get_stock_history("AAPL")
print(df.tail())

news = get_company_news("AAPL", "2024-12-01", "2024-12-31")
pprint(news[:2])

print("=== Done ===")