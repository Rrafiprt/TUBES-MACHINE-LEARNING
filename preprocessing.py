
import cv2
import numpy as np
import os
import pandas as pd

SPLIT_BASE = "data/split"
OUTPUT_BASE = "data/processed_v2"
FEATURES_FOLDER = "features"
SPLITS = ["train", "validation", "test"]
CATEGORIES = ["genuine", "forged"]
TARGET_SIZE = (300, 150)  # (width, height) -- ukuran seragam HANYA untuk gambar yang disimpan


def get_bounding_box_features(binary_full_size):
    """
    Hitung aspect_ratio dan ink_ratio dari BOUNDING BOX area tinta,
    memakai gambar biner UKURAN ASLI (sebelum resize).
    """
    coords = cv2.findNonZero(binary_full_size)  # koordinat semua piksel tinta (putih)

    if coords is None:
        # Tidak ada tinta terdeteksi (gambar kosong/gagal threshold)
        return {"aspect_ratio": 0, "ink_ratio": 0, "bbox_w": 0, "bbox_h": 0}

    x, y, w, h = cv2.boundingRect(coords)  # bounding box area tinta

    aspect_ratio = w / h if h > 0 else 0
    ink_pixels_in_bbox = np.sum(binary_full_size[y:y+h, x:x+w] == 255)
    bbox_area = w * h
    ink_ratio = ink_pixels_in_bbox / bbox_area if bbox_area > 0 else 0

    return {"aspect_ratio": aspect_ratio, "ink_ratio": ink_ratio, "bbox_w": w, "bbox_h": h}


def process_one_image(image_path):
    """
    Proses 1 gambar:
    - Load gambar asli (ukuran asli)
    - Convert grayscale
    - Binarization (Otsu) -- pada ukuran ASLI, untuk hitung bounding box features
    - Resize KEDUA versi (grayscale & binary) ke ukuran seragam untuk disimpan
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_full = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Fitur geometris dihitung dari ukuran ASLI (sebelum resize) -- ini perbaikannya
    bbox_features = get_bounding_box_features(binary_full)

    # Resize kedua versi untuk disimpan & dipakai fitur lain (GLCM, Hu Moments, statistik dasar)
    gray_resized = cv2.resize(gray_full, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    binary_resized = cv2.resize(binary_full, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    return gray_resized, binary_resized, bbox_features


def process_all_images():
    metadata_rows = []
    total_success = 0
    total_failed = 0

    for split_name in SPLITS:
        for category in CATEGORIES:
            input_folder = os.path.join(SPLIT_BASE, split_name, category)
            gray_out_folder = os.path.join(OUTPUT_BASE, split_name, category, "gray")
            binary_out_folder = os.path.join(OUTPUT_BASE, split_name, category, "binary")
            os.makedirs(gray_out_folder, exist_ok=True)
            os.makedirs(binary_out_folder, exist_ok=True)

            print(f"\n=== Memproses: {split_name}/{category} ===")

            if not os.path.exists(input_folder):
                print(f"  [PERINGATAN] Folder tidak ditemukan: {input_folder}")
                continue

            image_files = [f for f in os.listdir(input_folder)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            if not image_files:
                print(f"  [PERINGATAN] Tidak ada file gambar di {input_folder}")
                continue

            for filename in image_files:
                image_path = os.path.join(input_folder, filename)
                gray_resized, binary_resized, bbox_features = process_one_image(image_path)

                if gray_resized is None:
                    print(f"  [GAGAL] {filename}")
                    total_failed += 1
                    continue

                cv2.imwrite(os.path.join(gray_out_folder, filename), gray_resized)
                cv2.imwrite(os.path.join(binary_out_folder, filename), binary_resized)

                metadata_rows.append({
                    "split": split_name,
                    "category": category,
                    "filename": filename,
                    **bbox_features
                })
                total_success += 1

            print(f"  Selesai: {len(image_files)} file diproses")

    # Simpan metadata geometris ke CSV
    os.makedirs(FEATURES_FOLDER, exist_ok=True)
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = os.path.join(FEATURES_FOLDER, "geometric_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\n=== SELESAI ===")
    print(f"Total berhasil: {total_success}")
    print(f"Total gagal   : {total_failed}")
    print(f"Metadata geometris disimpan ke: {metadata_path}")


if __name__ == "__main__":
    process_all_images()