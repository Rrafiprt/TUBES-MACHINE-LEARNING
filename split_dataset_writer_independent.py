
import os
import re
import shutil
import random

RAW_FOLDERS = {
    "genuine": "data/raw/genuine",
    "forged": "data/raw/forged"
}
SPLIT_BASE = "data/split_writer_independent"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42


def extract_writer_id(filename):
    """
    Ambil writer ID dari nama file.
    Contoh: 'original_10_1.png' -> '10', 'forgeries_23_5.png' -> '23'
    """
    match = re.search(r'_(\d+)_', filename)
    return match.group(1) if match else None


def get_writer_to_files_mapping(folder_path):
    """
    Kelompokkan semua file dalam folder berdasarkan writer ID.
    Return: dict {writer_id: [list_of_filenames]}
    """
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    writer_map = {}
    unknown_count = 0

    for filename in files:
        writer_id = extract_writer_id(filename)
        if writer_id is None:
            unknown_count += 1
            continue
        writer_map.setdefault(writer_id, []).append(filename)

    if unknown_count > 0:
        print(f"  [PERINGATAN] {unknown_count} file tidak bisa dikenali writer ID-nya, dilewati")

    return writer_map


def split_writers(writer_ids):
    """
    Bagi daftar writer ID menjadi train/validation/test (80/10/10),
    berdasarkan JUMLAH WRITER, bukan jumlah gambar.
    """
    writer_ids = list(writer_ids)
    random.seed(RANDOM_SEED)
    random.shuffle(writer_ids)

    total = len(writer_ids)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    n_test = total - n_train - n_val

    train_writers = writer_ids[:n_train]
    val_writers = writer_ids[n_train:n_train + n_val]
    test_writers = writer_ids[n_train + n_val:]

    return {
        "train": train_writers,
        "validation": val_writers,
        "test": test_writers
    }


def copy_files_for_split(category, folder_path, writer_map, writer_split):
    """Copy semua file sesuai writer yang sudah dialokasikan ke masing-masing split."""
    for split_name, writer_list in writer_split.items():
        dest_folder = os.path.join(SPLIT_BASE, split_name, category)
        os.makedirs(dest_folder, exist_ok=True)

        file_count = 0
        for writer_id in writer_list:
            for filename in writer_map.get(writer_id, []):
                src_path = os.path.join(folder_path, filename)
                dest_path = os.path.join(dest_folder, filename)
                shutil.copy2(src_path, dest_path)
                file_count += 1

        print(f"  {split_name}: {len(writer_list)} writer -> {file_count} file -> {dest_folder}")


def process_category(category, folder_path):
    print(f"\n=== Memproses kategori: {category} ===")

    if not os.path.exists(folder_path):
        print(f"  [PERINGATAN] Folder tidak ditemukan: {folder_path}")
        return

    writer_map = get_writer_to_files_mapping(folder_path)
    writer_ids = list(writer_map.keys())

    print(f"  Jumlah writer unik ditemukan: {len(writer_ids)}")
    print(f"  Total file: {sum(len(v) for v in writer_map.values())}")

    writer_split = split_writers(writer_ids)

    print(f"  Pembagian writer -> train: {len(writer_split['train'])}, "
          f"validation: {len(writer_split['validation'])}, test: {len(writer_split['test'])}")

    copy_files_for_split(category, folder_path, writer_map, writer_split)

    return writer_split


def verify_no_overlap(all_writer_splits):
    """Pastikan tidak ada writer ID yang sama di lebih dari satu bagian (untuk kategori yang sama)."""
    print("\n=== VERIFIKASI: TIDAK ADA WRITER YANG TUMPANG TINDIH ===")
    for category, writer_split in all_writer_splits.items():
        train_set = set(writer_split["train"])
        val_set = set(writer_split["validation"])
        test_set = set(writer_split["test"])

        overlap_train_val = train_set & val_set
        overlap_train_test = train_set & test_set
        overlap_val_test = val_set & test_set

        if overlap_train_val or overlap_train_test or overlap_val_test:
            print(f"  [{category}] PERINGATAN: ada overlap writer! "
                  f"train-val: {overlap_train_val}, train-test: {overlap_train_test}, val-test: {overlap_val_test}")
        else:
            print(f"  [{category}] OK - tidak ada writer yang tumpang tindih antar split")


if __name__ == "__main__":
    all_writer_splits = {}

    for category, folder_path in RAW_FOLDERS.items():
        writer_split = process_category(category, folder_path)
        if writer_split:
            all_writer_splits[category] = writer_split

    verify_no_overlap(all_writer_splits)

    print("\n=== SELESAI ===")
    print(f"Dataset writer-independent disimpan di: {SPLIT_BASE}/")