"""Train XGBoost with grid search, log to MLflow, persist the best model."""
import os
import joblib
import pandas as pd
import mlflow
import xgboost as xgb

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

def find_project_root(marker="data/tourism.csv", start=None):
    start = os.path.abspath(start or os.path.dirname(__file__))
    current = start
    for _ in range(10):
        if os.path.exists(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(f"Could not locate {marker} from {start}")

REPO_ROOT  = find_project_root()
DEPLOY_DIR = os.path.join(REPO_ROOT, "deployment")
os.makedirs(DEPLOY_DIR, exist_ok=True)

MODEL_FILENAME = "best_tourism_package_model_v1.joblib"
MODEL_PATH     = os.path.join(DEPLOY_DIR, MODEL_FILENAME)

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "PitchSatisfactionScore", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Passport", "OwnCar", "Designation",
]
TARGET    = "ProdTaken"
THRESHOLD = 0.45

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tourism_prediction_prod")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), NUMERIC_FEATURES),
    (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
)
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, random_state=42, eval_metric="logloss",
)
model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators":      [100, 200],
    "xgbclassifier__max_depth":         [3, 5],
    "xgbclassifier__colsample_bytree":  [0.8, 1.0],
    "xgbclassifier__colsample_bylevel": [0.8, 1.0],
    "xgbclassifier__learning_rate":     [0.05, 0.1],
    "xgbclassifier__reg_lambda":        [1.0, 2.0],
}

with mlflow.start_run() as run:
    gs = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1,
                     scoring="roc_auc", verbose=1)
    gs.fit(Xtrain, ytrain)

    for i, params in enumerate(gs.cv_results_["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", gs.cv_results_["mean_test_score"][i])
            mlflow.log_metric("std_test_score",  gs.cv_results_["std_test_score"][i])

    mlflow.log_params(gs.best_params_)
    best_model = gs.best_estimator_

    y_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_test_proba  = best_model.predict_proba(Xtest)[:, 1]
    y_train_pred  = (y_train_proba >= THRESHOLD).astype(int)
    y_test_pred   = (y_test_proba  >= THRESHOLD).astype(int)

    tr = classification_report(ytrain, y_train_pred, output_dict=True)
    te = classification_report(ytest,  y_test_pred,  output_dict=True)

    metrics = {
        "threshold": THRESHOLD,
        "train_accuracy":  tr["accuracy"],
        "train_precision": tr["1"]["precision"],
        "train_recall":    tr["1"]["recall"],
        "train_f1":        tr["1"]["f1-score"],
        "test_accuracy":   te["accuracy"],
        "test_precision":  te["1"]["precision"],
        "test_recall":     te["1"]["recall"],
        "test_f1":         te["1"]["f1-score"],
        "test_roc_auc":    roc_auc_score(ytest, y_test_proba),
    }
    mlflow.log_metrics(metrics)

    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")

    print(f"\nModel saved to: {MODEL_PATH}")
    for k, v in metrics.items():
        print(f"  {k:>18}: {v:.4f}")
    print(f"\nMLflow run id: {run.info.run_id}")
