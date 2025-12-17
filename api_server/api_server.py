from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List

# 1. Kaydedilen Modelleri ve Encoder'ları Yükle
try:
    model = joblib.load("../lightGbm_model/signal_strength_model_lgbm.pkl")
    le_network = joblib.load("../lightGbm_model/network_encoder.pkl")
    le_timeofday = joblib.load("../lightGbm_model/timeofday_encoder.pkl")
    print("Modeller başarıyla yüklendi.")
except FileNotFoundError:
    print("HATA: Model veya encoder dosyaları bulunamadı. Lütfen dosya adlarını kontrol edin.")
    exit()

app = FastAPI(title="Predict Signal Strength Service")

# 2. Gelen Veri Yapısını Tanımlama (Pydantic Modeli)
# Bu, modelin beklediği 13 feature'ı listeler
class SignalFeatures(BaseModel):
    Latitude: float
    Longitude: float
    Signal_Quality_percent: float = 0.0  # % işareti kaldırılarak tanımlandı
    Data_Throughput_Mbps: float
    Latency_ms: float
    Hour: int
    DayOfWeek: int
    IsWeekend: int
    Network_Type: str
    BB60C_Measurement_dBm: float
    srsRAN_Measurement_dBm: float
    BladeRFxA9_Measurement_dBm: float
    TimeOfDay: str

# API'yi Test Etmek İçin Basit Bir Ana Sayfa
@app.get("/")
def read_root():
    return {"message": "Mobil Sinyal Gücü Tahmin Servisi Aktif"}


# 3. Tahmin Endpoint'i Oluşturma
@app.post("/predict_signal")
def predict_signal_strength(features: SignalFeatures):
    
    # Gelen veriyi Pandas DataFrame'e dönüştür
    input_data = features.dict()
    
    # 3.1. Encoder ile Kategorik Değişkenleri Sayısallaştırma
    try:
        # Network Type ve TimeOfDay'i yüklenen encoder'larla dönüştür
        input_data['Network_Type'] = le_network.transform([input_data['Network_Type']])[0]
        input_data['TimeOfDay'] = le_timeofday.transform([input_data['TimeOfDay']])[0]
    except ValueError as e:
        return {"error": f"Encoder hatası: Kategorik değerlerinizden biri bilinmiyor. {e}"}

    # Modelin beklediği tam feature listesi sırasına göre NumPy array oluştur
    feature_list = list(features.__fields__.keys())
    numerical_input = np.array([input_data[f] for f in feature_list])

    # 3.2. Tahmin Yapma
    prediction = model.predict(numerical_input.reshape(1, -1))[0]

    # 4. Sonucu Döndürme
    return {
        "predicted_signal_dBm": round(float(prediction), 2),
        "R2_Confidence": "0.895"
    }

# 4. Nasıl Çalıştırılır:
# Terminalde şu komutu çalıştırın: uvicorn api_server:app --reload
# Ardından tarayıcınızdan: http://127.0.0.1:8000/docs adresini ziyaret edin
# Orada /predict_signal endpoint'ini test edebilirsiniz.