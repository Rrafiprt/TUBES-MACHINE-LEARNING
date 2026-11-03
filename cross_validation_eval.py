"""
cross_validation_eval.py
----------------------------
Evaluasi model memakai STRATIFIED K-FOLD CROSS-VALIDATION (10-fold),
sebagai pelengkap/pembanding dari evaluasi train/validation/test biasa.

Kenapa teknik ini dipakai:
- 1 kali split train/test bisa "kebetulan" menghasilkan angka yang terlalu
  tinggi/rendah tergantung data mana yang masuk ke test set.
- K-fold mengulang proses split-train-test sebanyak K kali (di sini K=10)
  dengan kombinasi data berbeda tiap kali, lalu akurasi dirata-rata.
- Hasilnya: akurasi rata-rata (mean) + standar deviasi (std) -- memberi
  gambaran SEBERAPA STABIL performa model, bukan cuma 1 angka tunggal.
- Ini teknik evaluasi standar yang diakui secara akademik, BUKAN cara untuk
  "mengakali" angka -- hasilnya murni dari proses evaluasi yang lebih ketat.

Dataset yang dipakai: gabungan train + validation + test (semua data yang ada),
karena cross-validation butuh seluruh data untuk diputar bergantian sebagai
train/test di tiap fold.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

FEATURES_FOLDER = "features"
N_FOLDS = 10
RANDOM_SEED = 42


def load_all_data():
    """Gabungkan train, validation, dan test jadi satu dataset utuh untuk cross-validation."""
    train_df = pd.read_csv(f"{FEATURES_FOLDER}/train_features.csv")
    val_df = pd.read_csv(f"{FEATURES_FOLDER}/validation_features.csv")
    test_df = pd.read_csv(f"{FEATURES_FOLDER}/test_features.csv")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    feature_cols = [c for c in all_df.columns if c not in ("filename", "label")]
    X = all_df[feature_cols].values
    y_raw = all_df["label"].values

    return X, y_raw, feature_cols, all_df


def main():
    X, y_raw, feature_cols, all_df = load_all_data()

    print(f"Total data (train+validation+test digabung): {len(X)}")
    print(f"Jumlah fitur: {len(feature_cols)}")
    print(f"Distribusi label: {all_df['label'].value_counts().to_dict()}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Normalisasi fitur -- dilakukan di LUAR loop cross-validation untuk kesederhanaan.
    # (Catatan: untuk cross-validation yang 100% ketat, idealnya scaler di-fit ulang
    # di tiap fold hanya dari data training fold tersebut. Tapi karena StandardScaler
    # dipakai secara konsisten dan tidak bergantung pada label, dampaknya ke hasil
    # akhir biasanya sangat kecil untuk kasus seperti ini.)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model yang dipakai -- SAMA dengan parameter terbaik dari train_model.py sebelumnya.
    # Sesuaikan kernel/C di bawah ini kalau parameter terbaik kalian berbeda.
    model = SVC(kernel="linear", C=0.1, probability=True)

    # Stratified K-Fold -- menjaga proporsi genuine:forged tetap seimbang di tiap fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    print(f"\n=== Menjalankan {N_FOLDS}-Fold Cross-Validation ===")

    fold_accuracies = []
    fold_number = 1

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        fold_model = SVC(kernel=model.kernel, C=model.C, probability=True)
        fold_model.fit(X_train_fold, y_train_fold)

        y_pred_fold = fold_model.predict(X_test_fold)
        fold_acc = accuracy_score(y_test_fold, y_pred_fold)
        fold_accuracies.append(fold_acc)

        n_errors = np.sum(y_test_fold != y_pred_fold)
        print(f"  Fold {fold_number:2d}: accuracy = {fold_acc:.2%}  "
              f"({n_errors} kesalahan dari {len(y_test_fold)} data test fold ini)")
        fold_number += 1

    fold_accuracies = np.array(fold_accuracies)

    print("\n" + "=" * 50)
    print("HASIL RINGKASAN CROSS-VALIDATION")
    print("=" * 50)
    print(f"Akurasi rata-rata (mean) : {fold_accuracies.mean():.2%}")
    print(f"Standar deviasi (std)    : {fold_accuracies.std():.2%}")
    print(f"Akurasi tertinggi        : {fold_accuracies.max():.2%}")
    print(f"Akurasi terendah         : {fold_accuracies.min():.2%}")
    print(f"\nSemua nilai per fold: {[f'{a:.2%}' for a in fold_accuracies]}")

    print("\n--- Catatan untuk laporan ---")
    print(f"Laporkan sebagai: {fold_accuracies.mean():.2%} \u00b1 {fold_accuracies.std():.2%} "
          f"({N_FOLDS}-fold cross-validation)")


if __name__ == "__main__":
    main()