from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import numpy as np
import cv2
import io

# Import fungsi perhitungan spesifik dari file feature_extraction.py Anda
from feature_extraction import (
    extract_basic_stats, 
    extract_glcm_features, 
    extract_num_components, 
    extract_hu_moments
)

app = FastAPI()

# 1. Load Model, Scaler, dan Daftar Kolom Fitur
try:
    model = joblib.load("../models/signature_model.joblib")
    scaler = joblib.load("../models/signature_scaler.joblib")
    feature_cols = joblib.load("../models/signature_feature_cols.joblib")
    print("Berhasil memuat model, scaler, dan feature_cols!")
except Exception as e:
    print(f"Error memuat file .joblib: {e}")

@app.post("/predict")
async def predict_signature(file: UploadFile = File(...)):
    try:
        # 2. Baca file gambar yang diunggah
        contents = await file.read()
        
        # Konversi byte gambar menjadi array numpy (format yang dipakai OpenCV)
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar yang valid.")

        # 3. Buat citra Grayscale dan Biner (Preprocessing)
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Binarisasi menggunakan Otsu's Thresholding
        # Asumsi: Tanda tangan (tinta) berwarna gelap di atas kertas terang
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Untuk component extraction, OpenCV biasanya membaca objek (tinta) sebagai warna putih (255)
        # Jadi kita buat juga versi biner yang di-invert (tinta putih, latar hitam)
        _, binary_inv = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Ekstraksi Fitur dari gambar yang dikirim
        features_dict = {}
        features_dict.update(extract_basic_stats(gray_img))
        features_dict.update(extract_glcm_features(gray_img))
        features_dict.update(extract_num_components(binary_inv))
        features_dict.update(extract_hu_moments(binary_inv))
        
        # Ekstraksi Fitur Geometris (dihitung on-the-fly karena tidak ada metadata.csv)
        h, w = gray_img.shape
        features_dict["aspect_ratio"] = w / h if h != 0 else 0
        
        # Ink ratio: jumlah piksel tinta (hitam pada binary_img) dibagi total piksel
        ink_pixels = np.sum(binary_img == 0)
        features_dict["ink_ratio"] = ink_pixels / (h * w)

        # 5. Susun fitur ke dalam array sesuai urutan saat training
        # Ini sangat penting agar kolom fitur tidak tertukar
        raw_features = []
        for col in feature_cols:
            if col in features_dict:
                raw_features.append(features_dict[col])
            else:
                # Fallback jika ada kolom yang terlewat
                raw_features.append(0) 
                
        features_array = np.array(raw_features).reshape(1, -1)
        
        # 6. Normalisasi & Prediksi
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)
        
        # Tentukan Hasil (Sesuaikan "genuine" atau "forged" dengan label training Anda)
        hasil = "Asli" if prediction[0] == "genuine" else "Palsu"
        
        return {
            "status": "success",
            "prediction": hasil,
            "filename": file.filename
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}