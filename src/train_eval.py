import argparse, json, os
import pandas as pd, mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

p = argparse.ArgumentParser()
p.add_argument("--data"); p.add_argument("--model_out"); p.add_argument("--metrics_out")
p.add_argument("--n_estimators", type=int, default=100)
a = p.parse_args()

tr = pd.read_csv(f"{a.data}/train.csv"); te = pd.read_csv(f"{a.data}/test.csv")
model = RandomForestClassifier(n_estimators=a.n_estimators, random_state=42)
model.fit(tr.drop(columns="target"), tr["target"])
acc = accuracy_score(te["target"], model.predict(te.drop(columns="target")))

mlflow.log_param("n_estimators", a.n_estimators)
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.save_model(model, a.model_out, serialization_format="cloudpickle",
    extra_pip_requirements=["azureml-ai-monitoring", "azureml-inference-server-http"])
os.makedirs(a.metrics_out, exist_ok=True)
json.dump({"accuracy": acc}, open(f"{a.metrics_out}/metrics.json", "w"))
print("accuracy", acc)
