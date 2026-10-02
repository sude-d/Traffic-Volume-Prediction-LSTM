import requests
import random
API_URL = "http://127.0.0.1:8000/predict"
sample_sequence = []
for i in range(24):
    temp = random.uniform(280, 300)
    rain = 0
    snow = 0
    clouds = random.randint(0, 100)
    hour = i
    day_of_week = random.randint(0, 6)
    month = random.randint(1, 12)
    sample_sequence.append([
        temp,
        rain,
        snow,
        clouds,
        hour,
        day_of_week,
        month
    ])
response = requests.post(
    API_URL,
    json={"sequence": sample_sequence}
)
if response.status_code == 200:
    result = response.json()
    print("Predicted traffic volume:", result["predicted_traffic_volume"])
else:
    print("Status code:", response.status_code)
    print("Error:", response.text)