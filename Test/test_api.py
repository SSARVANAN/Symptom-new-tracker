import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "stomach_pain":0, "vomiting":1, "fatigue":1, "anxiety":0,
    "weight_loss":0, "restlessness":0, "lethargy":1, "cough":1,
    "high_fever":1, "breathlessness":0, "loss_of_appetite":0, "mild_fever":1,
    "malaise":1, "chest_pain":0, "fast_heart_rate":0, "obesity":0,
    "swollen_extremeties":0, "history_of_alcohol_consumption":0, "palpitations":0
}

response = requests.post(url, json=payload)
print(response.json())
