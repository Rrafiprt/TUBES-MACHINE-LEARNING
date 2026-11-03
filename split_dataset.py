
import os
import shutil
import random

# ---- KONFIGURASI ----
RAW_FOLDERS = {
    "genuine": "data/raw/genuine",
    "forged": "data/raw/forged"
}
SPLIT_BASE = "data/split"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42  # supaya hasil split selalu sama tiap kali dijalankan ulang (reproducible)


def split_category(category, folder_path):
    """
    Membagi semua file dalam satu kategori (genuine atau forged)
    menjadi train/validation/test sesuai rasio yang ditentukan.
    """
    print(f"\n=== Memproses kategori: {category} ===")

    # Ambil semua file gambar di folder
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not files:
        print(f"  [PERINGATAN] Tidak ada file gambar di {folder_path}")
        return

    total = len(files)
    print(f"  Total file ditemukan: {total}")

    # Acak urutan file supaya pembagian tidak bias urutan asli
    random.seed(RANDOM_SEED)
    random.shuffle(files)

    # Hitung jumlah untuk masing-masing bagian
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    # Sisanya masuk ke test, supaya totalnya pasti pas (menghindari selisih pembulatan)
    n_test = total - n_train - n_val

    print(f"  Pembagian -> train: {n_train}, validation: {n_val}, test: {n_test}")

    # Bagi list file sesuai jumlah di atas
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    splits = {
        "train": train_files,
        "validation": val_files,
        "test": test_files
    }

    # Copy file ke folder tujuan masing-masing
    for split_name, file_list in splits.items():
        dest_folder = os.path.join(SPLIT_BASE, split_name, category)
        os.makedirs(dest_folder, exist_ok=True)

        for filename in file_list:
            src_path = os.path.join(folder_path, filename)
            dest_path = os.path.join(dest_folder, filename)
            shutil.copy2(src_path, dest_path)

        print(f"  [OK] {len(file_list)} file disalin ke {dest_folder}")


def verify_split():
    """Menampilkan ringkasan jumlah file di setiap folder split sebagai konfirmasi akhir."""
    print("\n=== VERIFIKASI HASIL SPLIT ===")
    for split_name in ["train", "validation", "test"]:
        for category in RAW_FOLDERS.keys():
            folder = os.path.join(SPLIT_BASE, split_name, category)
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder)
                             if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                print(f"  {folder}: {count} file")
            else:
                print(f"  {folder}: [TIDAK DITEMUKAN]")


if __name__ == "__main__":
    for category, folder_path in RAW_FOLDERS.items():
        if not os.path.exists(folder_path):
            print(f"[PERINGATAN] Folder tidak ditemukan: {folder_path}")
            continue
        split_category(category, folder_path)

    verify_split()

    print("\n=== SELESAI ===")
    print("Dataset sudah dibagi ke data/split/train, data/split/validation, data/split/test")