import json
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tr.values, y_tr)

print("Test accuracy:", accuracy_score(y_te, model.predict(X_te.values)))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")

with open("model/training_stats.json", "w") as f:
    json.dump(
        {
            "features": list(X.columns),
            "mean": X_tr.mean().tolist(),
            "std": X_tr.std().tolist(),
        },
        f,
    )