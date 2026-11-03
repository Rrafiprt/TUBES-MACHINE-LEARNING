"""
train_model.py (versi menampilkan akurasi TRAIN, VALIDATION, dan TEST sekaligus)
-------------------------------------------------------------------------------
1. Load train_features.csv, validation_features.csv, test_features.csv
2. Coba beberapa kombinasi parameter SVM, pilih yang terbaik berdasarkan VALIDATION set
3. Latih ulang model terbaik
4. Tampilkan akurasi di KETIGA bagian: train, validation, dan test
5. Simpan model final
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES_FOLDER = "features"
MODELS_FOLDER = "models"


def load_split(name):
    path = f"{FEATURES_FOLDER}/{name}_features.csv"
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("filename", "label")]
    X = df[feature_cols].values
    y_raw = df["label"].values
    return X, y_raw, feature_cols, df


def main():
    # ---- 1. Load semua data ----
    X_train, y_train_raw, feature_cols, _ = load_split("train")
    X_val, y_val_raw, _, _ = load_split("validation")
    X_test, y_test_raw, _, _ = load_split("test")

    print(f"Jumlah data -> train: {len(X_train)}, validation: {len(X_val)}, test: {len(X_test)}")

    # ---- 2. Encode label ----
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ---- 3. Normalisasi fitur ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ---- 4. Tuning parameter di VALIDATION set ----
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

    # ---- 5. Latih model final dengan gabungan train+validation ----
    X_trainval_scaled = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])

    final_model = SVC(**best_params, probability=True)
    final_model.fit(X_trainval_scaled, y_trainval)

    # ---- 6. AKURASI DI KETIGA BAGIAN (menggunakan model final) ----
    print("\n" + "=" * 50)
    print("RINGKASAN AKURASI: TRAIN vs VALIDATION vs TEST")
    print("=" * 50)

    # Train accuracy (seberapa baik model "menghafal" data training)
    train_pred = final_model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\n[TRAIN]      Accuracy: {train_acc:.2%}  (n={len(y_train)})")

    # Validation accuracy (dipakai untuk pilih parameter tadi)
    val_pred = final_model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"[VALIDATION] Accuracy: {val_acc:.2%}  (n={len(y_val)})")

    # Test accuracy (evaluasi final, paling penting untuk dilaporkan)
    test_pred = final_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"[TEST]       Accuracy: {test_acc:.2%}  (n={len(y_test)})")

    print("\n--- Detail evaluasi TEST SET (untuk laporan) ---")
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, test_pred))
    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_pred, target_names=le.classes_, zero_division=0))

    # ---- 7. Peringatan otomatis kalau ada indikasi overfitting/underfitting ----
    print("\n--- Catatan otomatis ---")
    gap_train_test = train_acc - test_acc
    if gap_train_test > 0.10:
        print(f"[PERHATIAN] Selisih train-test cukup besar ({gap_train_test:.2%}) -> indikasi OVERFITTING")
    elif train_acc < 0.70:
        print(f"[PERHATIAN] Akurasi train rendah ({train_acc:.2%}) -> indikasi UNDERFITTING")
    else:
        print("Tidak ada indikasi overfitting/underfitting yang mencolok (selisih train-test wajar).")

    # ---- 8. Simpan model final ----
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    joblib.dump(final_model, f"{MODELS_FOLDER}/signature_model.joblib")
    joblib.dump(scaler, f"{MODELS_FOLDER}/signature_scaler.joblib")
    joblib.dump(le, f"{MODELS_FOLDER}/signature_label_encoder.joblib")
    joblib.dump(feature_cols, f"{MODELS_FOLDER}/signature_feature_cols.joblib")

    print(f"\nModel final disimpan ke folder '{MODELS_FOLDER}/'")


if __name__ == "__main__":
    main()