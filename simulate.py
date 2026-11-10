import random, sys, time
import requests

URL = "http://localhost:8000"
CLASSES = ["setosa", "versicolor", "virginica"]
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"

def sample(drift=False):
    s = {
        "sepal_length": random.gauss(5.8, 0.8),
        "sepal_width": random.gauss(3.0, 0.4),
        "petal_length": random.gauss(3.8, 1.7),
        "petal_width": random.gauss(1.2, 0.7)
    }
    if drift:
        s["petal_length"] += 2.0
        s["sepal_length"] += 1.0
    return s

while True:
    if mode == "errors" and random.random() < 0.3:
        requests.post(f"{URL}/predict", json={"bad": 1})
    else:
        r = requests.post(f"{URL}/predict", json=sample(mode == "drift"))
        if r.ok and random.random() < 0.3:
            pred = r.json()["prediction"]
            p_correct = 0.95 if mode == "normal" else 0.6
            actual = pred if random.random() < p_correct else random.choice(CLASSES)
            requests.post(f"{URL}/feedback", json={"predicted": pred, "actual": actual})
    time.sleep(random.uniform(0.05, 0.3))