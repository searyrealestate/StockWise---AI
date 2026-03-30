from api_manager import APIManager

manager = APIManager()

for _ in range(3):
    name, key = manager.get_available_api()
    print("Using API:", name, "| Key:", key)

print("Today Usage:", manager.get_usage_summary())
