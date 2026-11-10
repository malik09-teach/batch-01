from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data():
    data_path = Path(__file__).resolve().parent.parent / "classification" / "heart_disease_uci.csv"
    data = pd.read_csv(data_path)
    if "num" not in data.columns:
        raise ValueError("Expected the target column 'num' in the heart-disease dataset.")
    return data, data_path


def main():
    data, data_path = load_data()
    data = data.dropna(subset=["num"]).copy()
    data["target"] = (data["num"] > 0).astype(int)

    excluded_columns = {"id", "dataset", "num", "target"}
    feature_columns = [column for column in data.columns if column not in excluded_columns]
    if not feature_columns:
        raise ValueError("No usable feature columns were found.")

    X = data[feature_columns].copy()
    y = data["target"]
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    # CatBoost accepts missing numeric values, but categorical values must be non-null.
    for column in categorical_features:
        X[column] = X[column].fillna("__missing__").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train, cat_features=categorical_features)

    predictions = model.predict(X_test).reshape(-1).astype(int)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("CatBoost Heart-Disease Classification Results")
    print(f"Dataset: {data_path}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No disease", "Disease"],
            zero_division=0,
        )
    )

    importance = (
        pd.Series(model.feature_importances_, index=feature_columns)
        .sort_values(ascending=False)
        .head(10)
    )
    print("Top feature importances:")
    print(importance.to_string())
    print("\nEducational example only; do not use for medical diagnosis.")


if __name__ == "__main__":
    main()
