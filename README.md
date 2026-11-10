# MLOps Model Monitoring Stack

An end-to-end MLOps monitoring solution for an Iris classification model using FastAPI, Prometheus, and Grafana. The system continuously tracks service health metrics, model performance, and real-time feature drift against baseline training data using Docker Compose.

---

##  Repository Structure

```text
.
├── app.py                # FastAPI application exposing model endpoints & /metrics
├── simulate.py           # Traffic generator (normal, drift, errors)
├── alert_rules.yml       # Prometheus alerting rules (drift, errors, outage, latency)
├── prometheus.yml        # Prometheus server configuration
├── docker-compose.yml    # Container orchestration for all services
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── grafana/
    └── dashboard.json    # Exported 9-panel Grafana dashboard JSON

```

---

## 🛠️ System Architecture & Services

* **Model Service (`app.py`):** FastAPI app serving Iris flower predictions via `POST /predict`, calculating real-time feature $z$-scores in a rolling window, and exposing operational metrics at `/metrics`.
* **Traffic Simulator (`simulate.py`):** Client script used to generate realistic synthetic traffic scenarios (baseline, feature drift, API error injection).
* **Prometheus:** Scrapes `/metrics` every 5 seconds and evaluates alerting rules configured in `alert_rules.yml`.
* **Grafana:** Visualizes operational, statistical, and infrastructure metrics across a pre-configured 9-panel dashboard.

---

##  Quick Start Guide

### Prerequisites

* [Docker Desktop](https://www.docker.com/) installed and running
* Python 3.9+ installed locally

### 1. Launch the Stack

Spin up the Model Service, Prometheus, Grafana, and Node Exporter containers:

```bash
docker compose up --build -d

```

Verify service endpoints:

* **FastAPI Docs:** `http://localhost:8000/docs`
* **Prometheus UI:** `http://localhost:9090`
* **Grafana Dashboard:** `http://localhost:3000` *(Default login: `admin` / `admin`)*

### 2. Install Client Dependencies

Install Python packages required to run the traffic simulator:

```bash
pip install -r requirements.txt

```

---

##  Simulation Scenarios

Run the following traffic scenarios to observe dashboard metrics and validate Prometheus alerts:

### Scenario A: Normal Traffic (Baseline)

Streams clean requests from the training dataset distribution to establish baseline operational metrics.

```bash
python simulate.py normal

```

### Scenario B: Feature Drift

Injects synthetic feature shift (`petal_length += 2.0`, `sepal_length += 1.0`).

```bash
python simulate.py drift

```

* **Expected Result:** $Z$-scores spike above $+3$ on Grafana Panel 6; `FeatureDriftDetected` alert turns **FIRING** on Prometheus (`http://localhost:9090/alerts`).

### Scenario C: High Error Rate

Triggers server-side exceptions and unhandled input errors on the `/predict` endpoint.

```bash
python simulate.py errors

```

* **Expected Result:** Error rate panel spikes above 5%; `HighErrorRate` alert turns **FIRING**.

### Scenario D: Service Outage

Simulates a complete service failure by stopping the model container.

```bash
docker compose stop model-service

```

* **Expected Result:** Service Status panel switches to **DOWN** (Red); `ModelServiceDown` alert turns **FIRING** after 30 seconds.

### Recovery

Restart the service container and return to normal traffic to resolve all alerts back to **Inactive**:

```bash
docker compose start model-service
python simulate.py normal

```

---

##  Grafana Dashboard Panels

The custom monitoring dashboard is organized into 9 dedicated panels:

1. **Request rate (req/s)** – Time series of incoming throughput across endpoints.
2. **Error Rate (%)** – Stat gauge showing percentage of 5xx responses.
3. **p95 Latency (s)** – 95th percentile response latency trend.
4. **Predictions by Class** – Distribution of model outputs (Setosa, Versicolor, Virginica).
5. **Average Confidence** – Gauge showing rolling mean prediction probability.
6. **Feature Drift Z-Score** – Real-time feature shift tracking against baseline mean and std dev.
7. **Rolling Accuracy (5 min)** – Gauge monitoring model performance over a 5-minute window.
8. **Service Status** – Stat panel displaying real-time uptime status (**UP** / **DOWN**).
9. **Host CPU Usage (%)** – System resource consumption metric.

---

##  Alert Rules Overview

Configured alert thresholds in `alert_rules.yml`:

| Alert Name | Condition | Pending Duration | Severity |
| --- | --- | --- | --- |
| `ModelServiceDown` | `up{job="model-service"} == 0` | 30s | Critical |
| `HighErrorRate` | 5xx HTTP Error Rate > 5% | 1m | Critical |
| `HighP95Latency` | p95 Latency > 0.5s | 1m | Warning |
| `FeatureDriftDetected` | $\text{abs}(z\text{-score}) > 3$ | 1m | Warning |
| `LowModelAccuracy` | Rolling Accuracy (5m) < 85% | 1m | Critical |
