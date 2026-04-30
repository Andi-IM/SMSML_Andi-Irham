import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import argparse
import os
import dagshub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
REPO_OWNER = 'Andi-IM'
REPO_NAME = 'SMSML_Andi-Irham'
LOCAL_URI = "http://0.0.0.0:5000"
EXPERIMENT_NAME = "Healthcare CVE Classification"


def train_model(params, data_vars):
    """Train the RandomForest model. Pure ML logic, no MLflow dependency."""
    X_train, X_test, y_train, y_test, df = data_vars

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return model, y_pred, accuracy, f1_macro


def log_to_mlflow(model, y_pred, accuracy, f1_macro, params, data_vars, suffix=""):
    """Log params, metrics, plots, and model to the CURRENT active MLflow run."""
    X_train, X_test, y_train, y_test, df = data_vars

    # Log Hyperparameters
    mlflow.log_params(params)

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1_macro)

    # --- MANUAL LOGGING ---
    mlflow.set_tags({
        "project": "Healthcare-CVE",
        "environment": "Workflow-CI",
        "run_type": suffix
    })

    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title(f"Top 10 Features ({suffix})")
    plt.tight_layout()
    fi_path = f"feat_imp_{suffix}.png"
    plt.savefig(fi_path)
    mlflow.log_artifact(fi_path, artifact_path="plots")
    plt.close()
    if os.path.exists(fi_path): os.remove(fi_path)

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix ({suffix})")
    cm_path = f"cm_{suffix}.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, artifact_path="plots")
    plt.close()
    if os.path.exists(cm_path): os.remove(cm_path)

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="healthcare_cybersecurity_preprocessing.csv")
    args = parser.parse_args()

    RANDOM_STATE = 42

    # Load and Split Data
    df = pd.read_csv(args.dataset)
    target_col = "CVSS_Risk_Bin"
    drop_cols = ["CVE_ID", "CVSS_Score_Scaled", "Severity_CRITICAL", "Severity_HIGH",
                 "Severity_LOW", "Severity_MEDIUM", "Severity_UNKNOWN"]

    X = df.drop(columns=[target_col, *drop_cols], errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": RANDOM_STATE,
        "dataset": args.dataset
    }
    data_vars = (X_train, X_test, y_train, y_test, df)

    # === Train model ONCE ===
    print("Training model...")
    model, y_pred, accuracy, f1_macro = train_model(params, data_vars)
    print(f"Results: accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}")

    # Detect if running inside `mlflow run .` (CI mode)
    active_run_id = os.environ.get("MLFLOW_RUN_ID")

    if active_run_id:
        # =============================================
        # CI MODE: Running inside `mlflow run .`
        # Log directly to the outer run so artifacts
        # are available for `mlflow models build-docker`
        # =============================================
        print(f"\nCI Mode: Logging to active run {active_run_id}...")
        with mlflow.start_run(run_id=active_run_id):
            log_to_mlflow(model, y_pred, accuracy, f1_macro, params, data_vars, "CI")
        print("CI logging successful!")

        # Replicate to DagsHub (best-effort, non-blocking)
        if os.environ.get("DAGSHUB_USER_TOKEN"):
            print("\nReplicating to DagsHub...")
            try:
                # Remove local experiment ID — it doesn't exist on DagsHub
                saved_exp_id = os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
                dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
                mlflow.set_experiment(EXPERIMENT_NAME)
                with mlflow.start_run(run_name="RF_Workflow_DagsHub"):
                    log_to_mlflow(model, y_pred, accuracy, f1_macro, params, data_vars, "DagsHub")
                print("DagsHub replication successful!")
            except Exception as e:
                print(f"DagsHub replication failed (non-critical): {e}")
            finally:
                # Restore experiment ID if other steps need it
                if saved_exp_id:
                    os.environ["MLFLOW_EXPERIMENT_ID"] = saved_exp_id
    else:
        # =============================================
        # STANDALONE MODE: Local development
        # =============================================

        # 1. Log to Local MLflow server
        print("\nLogging to Local MLflow...")
        try:
            mlflow.set_tracking_uri(LOCAL_URI)
            try:
                mlflow.set_experiment(EXPERIMENT_NAME)
            except:
                pass
            with mlflow.start_run(run_name="RF_Workflow_Local"):
                log_to_mlflow(model, y_pred, accuracy, f1_macro, params, data_vars, "Local")
            print("Local logging successful!")
        except Exception as e:
            print(f"Local logging failed: {e}")

        # 2. Log to DagsHub
        print("\nLogging to DagsHub MLflow...")
        try:
            dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="RF_Workflow_DagsHub"):
                log_to_mlflow(model, y_pred, accuracy, f1_macro, params, data_vars, "DagsHub")
            print("DagsHub logging successful!")
        except Exception as e:
            print(f"DagsHub logging failed: {e}")


if __name__ == "__main__":
    main()
