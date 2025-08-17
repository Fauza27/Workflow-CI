import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import time
import sys

# ========================================
# Load data preprocessing
# ========================================
# Gunakan path relatif yang dikirim MLflow Project
if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = "landmine_preprocessing.csv"

df = pd.read_csv(data_path)

X = df.drop("Mine type", axis=1)
y = df["Mine type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================================
# Setup MLflow lokal
# ========================================
mlflow.set_tracking_uri("file:///home/runner/work/Workflow-CI/Workflow-CI/mlruns")
mlflow.set_experiment("Mine Classification with RF")

# ========================================
# Hyperparameter Tuning
# ========================================
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

# ========================================
# Training + Logging Manual
# ========================================
with mlflow.start_run():
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("precision_macro", precision_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("recall_macro", recall_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("matthews_corrcoef", matthews_corrcoef(y_test, y_pred))
    mlflow.log_metric("train_time_sec", train_time)
    mlflow.log_metric("test_size_ratio", len(X_test)/len(X))

    # Log model
    mlflow.sklearn.log_model(best_model, "random_forest_model")

print("Best params:", grid_search.best_params_)
