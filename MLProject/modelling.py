import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import argparse
import os
import dagshub

mlflow.autolog()  # Mandatory autologging


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




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="healthcare_cybersecurity_preprocessing/healthcare_cybersecurity_preprocessing.csv")
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

    # Detect if running inside `mlflow run .` (CI mode)
    active_run_id = os.environ.get("MLFLOW_RUN_ID")

    if active_run_id:
        print(f"\nCI Mode: Logging to active run {active_run_id}...")
        with mlflow.start_run(run_id=active_run_id):
            model, y_pred, accuracy, f1_macro = train_model(params, data_vars)
        print(f"CI Results: accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}")

        if os.environ.get("DAGSHUB_USER_TOKEN"):
            print("\nReplicating to DagsHub...")
            try:
                saved_exp_id = os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
                dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
                mlflow.set_experiment(EXPERIMENT_NAME)
                with mlflow.start_run(run_name="RF_Workflow_DagsHub"):
                    train_model(params, data_vars)
                print("DagsHub replication successful!")
            except Exception as e:
                print(f"DagsHub replication failed (non-critical): {e}")
            finally:
                if saved_exp_id:
                    os.environ["MLFLOW_EXPERIMENT_ID"] = saved_exp_id
    else:
        # STANDALONE MODE
        print("\nLogging to Local MLflow...")
        try:
            mlflow.set_tracking_uri(LOCAL_URI)
            try:
                mlflow.set_experiment(EXPERIMENT_NAME)
            except:
                pass
            with mlflow.start_run(run_name="RF_Workflow_Local"):
                model, y_pred, accuracy, f1_macro = train_model(params, data_vars)
            print(f"Local Results: accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}")
        except Exception as e:
            print(f"Local logging failed: {e}")

        print("\nLogging to DagsHub MLflow...")
        try:
            dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="RF_Workflow_DagsHub"):
                train_model(params, data_vars)
            print("DagsHub logging successful!")
        except Exception as e:
            print(f"DagsHub logging failed: {e}")


if __name__ == "__main__":
    main()
