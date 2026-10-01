import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = "breast_cancer.csv"


def train_model():

    print("Loading breast cancer dataset...")

    data = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print("Number of columns:", len(data.columns))

    X = data.drop("target", axis=1)
    y = data["target"]

    print("Number of features:", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])

    print("Training model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(
        model,
        "breast_cancer_model.pkl"
    )

    print("\nModel saved as breast_cancer_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "number_of_features": X.shape[1]
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")

    return accuracy


if __name__ == "__main__":
    train_model()
