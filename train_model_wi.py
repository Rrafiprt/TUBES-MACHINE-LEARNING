
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES_FOLDER = "features_wi"
MODELS_FOLDER = "models_wi"


def load_split(name):
    path = f"{FEATURES_FOLDER}/{name}_features.csv"
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("filename", "label")]
    X = df[feature_cols].values
    y_raw = df["label"].values
    return X, y_raw, feature_cols, df


def main():
    X_train, y_train_raw, feature_cols, _ = load_split("train")
    X_val, y_val_raw, _, _ = load_split("validation")
    X_test, y_test_raw, _, _ = load_split("test")

    print(f"Jumlah data -> train: {len(X_train)}, validation: {len(X_val)}, test: {len(X_test)}")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    param_grid = [
        {"kernel": "linear", "C": 0.1},
        {"kernel": "linear", "C": 1.0},
        {"kernel": "linear", "C": 10.0},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        {"kernel": "poly", "C": 1.0, "degree": 3},
    ]

    print("\n=== Tuning parameter menggunakan VALIDATION set ===")
    best_score = -1
    best_params = None

    for params in param_grid:
        model = SVC(**params)
        model.fit(X_train_scaled, y_train)
        val_pred = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"  Params: {params} -> Validation accuracy: {val_acc:.2%}")
        if val_acc > best_score:
            best_score = val_acc
            best_params = params

    print(f"\nParameter terbaik: {best_params} (validation accuracy: {best_score:.2%})")

    X_trainval_scaled = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])

    final_model = SVC(**best_params, probability=True)
    final_model.fit(X_trainval_scaled, y_trainval)

    print("\n=== EVALUASI FINAL (TEST SET - WRITER INDEPENDENT) ===")
    test_pred = final_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Test Accuracy: {test_acc:.2%}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=le.classes_, zero_division=0))

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    joblib.dump(final_model, f"{MODELS_FOLDER}/signature_model.joblib")
    joblib.dump(scaler, f"{MODELS_FOLDER}/signature_scaler.joblib")
    joblib.dump(le, f"{MODELS_FOLDER}/signature_label_encoder.joblib")
    joblib.dump(feature_cols, f"{MODELS_FOLDER}/signature_feature_cols.joblib")

    print(f"\nModel final disimpan ke folder '{MODELS_FOLDER}/'")


if __name__ == "__main__":
    main()