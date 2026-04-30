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
LOCAL_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Healthcare CVE Classification"

def run_training(tracking_uri, params, data_vars, run_name_suffix=""):
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except:
        pass

    X_train, X_test, y_train, y_test, df = data_vars

    with mlflow.start_run(run_name=f"RF_Workflow_{run_name_suffix}"):
        # Log Hyperparameters
        mlflow.log_params(params)

        # Train Model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        # --- MANUAL LOGGING ---
        mlflow.set_tags({
            "project": "Healthcare-CVE",
            "environment": "Workflow-CI",
            "run_type": run_name_suffix
        })

        # Feature Importance Plot
        plt.figure(figsize=(10, 6))
        feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
        plt.title(f"Top 10 Features ({run_name_suffix})")
        plt.tight_layout()
        fi_path = f"feat_imp_{run_name_suffix}.png"
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path, artifact_path="plots")
        plt.close()
        if os.path.exists(fi_path): os.remove(fi_path)

        # Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix ({run_name_suffix})")
        cm_path = f"cm_{run_name_suffix}.png"
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
        return accuracy, f1_macro

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

    # 1. Log to Local
    print("Logging to Local MLflow...")
    try:
        run_training(LOCAL_URI, params, data_vars, "Local")
    except Exception as e:
        print(f"Local logging failed: {e}")

    # 2. Log to DagsHub
    print("\nLogging to DagsHub MLflow...")
    try:
        dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
        run_training(mlflow.get_tracking_uri(), params, data_vars, "DagsHub")
    except Exception as e:
        print(f"DagsHub logging failed: {e}")

if __name__ == "__main__":
    main()
