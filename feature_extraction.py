
import cv2
import numpy as np
import pandas as pd
import os
from scipy import stats
from skimage.feature import graycomatrix, graycoprops

PROCESSED_BASE = "data/processed_v2"
FEATURES_FOLDER = "features"
SPLITS = ["train", "validation", "test"]
CATEGORIES = ["genuine", "forged"]

# GLCM dihitung di 4 arah, hasilnya dirata-rata -- lebih representatif
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]


def extract_basic_stats(gray_img):
    flat = gray_img.flatten()
    return {
        "mean": np.mean(flat), "std": np.std(flat), "variance": np.var(flat),
        "skewness": stats.skew(flat), "kurtosis": stats.kurtosis(flat),
        "entropy": stats.entropy(np.histogram(flat, bins=256)[0] + 1e-9)
    }


def extract_glcm_features(gray_img):
    """GLCM dari gambar GRAYSCALE, 4 arah dirata-rata."""
    glcm = graycomatrix(gray_img, distances=[1], angles=GLCM_ANGLES,
                          levels=256, symmetric=True, normed=True)
    return {
        "contrast": np.mean(graycoprops(glcm, "contrast")),
        "correlation": np.mean(graycoprops(glcm, "correlation")),
        "energy": np.mean(graycoprops(glcm, "energy")),
        "homogeneity": np.mean(graycoprops(glcm, "homogeneity"))
    }


def extract_num_components(binary_img):
    num_labels, _ = cv2.connectedComponents(binary_img)
    return {"num_components": num_labels - 1}


def extract_hu_moments(binary_img):
    """Hu Moments dihitung dari gambar biner (bentuk keseluruhan tanda tangan)."""
    moments = cv2.moments(binary_img)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return {f"hu_{i+1}": hu_log[i] for i in range(7)}


def process_split(split_name, geo_metadata):
    all_rows = []

    for category in CATEGORIES:
        gray_folder = os.path.join(PROCESSED_BASE, split_name, category, "gray")
        binary_folder = os.path.join(PROCESSED_BASE, split_name, category, "binary")

        print(f"\n=== Ekstraksi fitur: {split_name}/{category} ===")

        if not os.path.exists(gray_folder):
            print(f"  [PERINGATAN] Folder tidak ditemukan: {gray_folder}")
            continue

        image_files = [f for f in os.listdir(gray_folder)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        for filename in image_files:
            gray_img = cv2.imread(os.path.join(gray_folder, filename), cv2.IMREAD_GRAYSCALE)
            binary_img = cv2.imread(os.path.join(binary_folder, filename), cv2.IMREAD_GRAYSCALE)

            if gray_img is None or binary_img is None:
                print(f"  [GAGAL] {filename}")
                continue

            features = {}
            features.update(extract_basic_stats(gray_img))
            features.update(extract_glcm_features(gray_img))
            features.update(extract_num_components(binary_img))
            features.update(extract_hu_moments(binary_img))

            # Ambil aspect_ratio & ink_ratio dari metadata (dihitung dari ukuran ASLI)
            meta_row = geo_metadata[
                (geo_metadata["split"] == split_name) &
                (geo_metadata["category"] == category) &
                (geo_metadata["filename"] == filename)
            ]
            if not meta_row.empty:
                features["aspect_ratio"] = meta_row.iloc[0]["aspect_ratio"]
                features["ink_ratio"] = meta_row.iloc[0]["ink_ratio"]
            else:
                features["aspect_ratio"] = 0
                features["ink_ratio"] = 0

            features["filename"] = filename
            features["label"] = category
            all_rows.append(features)

        print(f"  Selesai: {len(image_files)} file")

    if not all_rows:
        print(f"[PERINGATAN] Tidak ada data untuk split '{split_name}'")
        return

    df = pd.DataFrame(all_rows)
    cols = ["filename", "label"] + [c for c in df.columns if c not in ("filename", "label")]
    df = df[cols]

    output_path = os.path.join(FEATURES_FOLDER, f"{split_name}_features.csv")
    df.to_csv(output_path, index=False)
    print(f"\n[OK] {split_name}_features.csv disimpan ({len(df)} baris) -> {output_path}")


if __name__ == "__main__":
    geo_metadata = pd.read_csv(os.path.join(FEATURES_FOLDER, "geometric_metadata.csv"))

    for split_name in SPLITS:
        process_split(split_name, geo_metadata)

    print("\n=== SEMUA SPLIT SELESAI DIPROSES ===")