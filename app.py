import collections, json, time
import joblib, numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
)

MODEL_VERSION = "1.0.0"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

model = joblib.load("model/model.joblib")
with open("model/training_stats.json") as f:
    stats = json.load(f)

FEATURES = stats["features"]
WINDOW = collections.deque(maxlen=100)

# Prometheus Metrics
REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["endpoint"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["predicted_class"])
CONFIDENCE = Histogram("model_prediction_confidence", "Top class probability", buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
DRIFT = Gauge("model_feature_drift_zscore", "Z-score of rolling feature mean vs training mean", ["feature"])
FEEDBACK = Counter("model_feedback_total", "Ground-truth feedback", ["outcome"])
ERRORS = Counter("model_inference_errors_total", "Failed predictions")
INFO = Gauge("model_info", "Serving model metadata", ["version"])
INFO.labels(MODEL_VERSION).set(1)

app = FastAPI(title="Iris Model Service")

class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Feedback(BaseModel):
    predicted: str
    actual: str

@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = route.path if route else "unmatched"
    LATENCY.labels(path).observe(time.perf_counter() - start)
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    return response

def update_drift():
    if len(WINDOW) < 20:
        return
    data = np.array(WINDOW)
    n = len(data)
    for i, name in enumerate(FEATURES):
        se = stats["std"][i] / np.sqrt(n)
        DRIFT.labels(name).set(float((data[:, i].mean() - stats["mean"][i]) / se))

@app.post("/predict")
def predict(f: Features):
    x = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]])
    try:
        proba = model.predict_proba(x)[0]
    except Exception:
        ERRORS.inc()
        raise HTTPException(status_code=500, detail="Prediction failed")
    
    label = int(np.argmax(proba))
    PREDICTIONS.labels(CLASS_NAMES[label]).inc()
    CONFIDENCE.observe(float(proba.max()))
    WINDOW.append(x[0])
    update_drift()
    return {
        "prediction": CLASS_NAMES[label],
        "confidence": round(float(proba.max()), 4),
        "model_version": MODEL_VERSION
    }

@app.post("/feedback")
def feedback(fb: Feedback):
    FEEDBACK.labels("correct" if fb.predicted == fb.actual else "incorrect").inc()
    return {"status": "recorded"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)