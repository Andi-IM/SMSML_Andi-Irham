import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="healthcare_cybersecurity_preprocessing.csv")
    args = parser.parse_args()

    RANDOM_STATE = 42
    N_ESTIMATORS = args.n_estimators
    MAX_DEPTH = args.max_depth
    DATASET = args.dataset

    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    try:
        mlflow.set_experiment("Healthcare CVE Classification")
    except Exception:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Healthcare CVE Classification")

    df = pd.read_csv(DATASET)

    target_col = "CVSS_Risk_Bin"
    drop_cols = [
        "CVE_ID",
        "CVSS_Score_Scaled",
        "Severity_CRITICAL",
        "Severity_HIGH",
        "Severity_LOW",
        "Severity_MEDIUM",
        "Severity_UNKNOWN",
    ]

    X = df.drop(columns=[target_col, *drop_cols], errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    input_example = X_train.iloc[:5]

    with mlflow.start_run():
        mlflow.log_params(
            {
                "n_estimators": N_ESTIMATORS,
                "max_depth": MAX_DEPTH,
                "random_state": RANDOM_STATE,
            }
        )

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        print({"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH})
        print({"test_accuracy": accuracy, "test_f1_macro": f1_macro})

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
        )

if __name__ == "__main__":
    main()
