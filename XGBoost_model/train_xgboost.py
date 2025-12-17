# # =========================================
# # 🎯 train_xgboost.py
# # Amaç: processed_data.csv verisiyle XGBoost modelini eğitmek
# # =========================================

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_absolute_error, r2_score
# from xgboost import XGBRegressor 
# import joblib
# import numpy as np

# # 1️⃣ Veri Yükleme
# data = pd.read_csv("../data/processed_data.csv")

# # 2️⃣ Hedef ve özellikleri belirle
# target = "Signal Strength (dBm)"
# features = [
#     "Latitude",
#     "Longitude",
#     "Signal Quality (%)",
#     "Data Throughput (Mbps)",
#     "Latency (ms)",
#     "Hour",
#     "DayOfWeek",
#     "IsWeekend",
#     "Network Type", 
#     "BB60C Measurement (dBm)",
#     "srsRAN Measurement (dBm)",
#     "BladeRFxA9 Measurement (dBm)",
#     "TimeOfDay"
# ]

# # 3️⃣ Kategorik Özellikleri encode et (Aynı Encoder'lar)
# le_network = LabelEncoder()
# data["Network Type"] = le_network.fit_transform(data["Network Type"])

# le_timeofday = LabelEncoder()
# data["TimeOfDay"] = le_timeofday.fit_transform(data["TimeOfDay"])


# # 4️⃣ Girdi ve hedef ayrımı
# X = data[features]
# y = data[target]

# # 5️⃣ Eğitim / test seti bölünmesi
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 6️⃣ Model oluşturma ve eğitme
# print("🚀 XGBoost modeli eğitiliyor...")
# # Temel XGBoost parametreleri
# model_xgb = XGBRegressor(
#     n_estimators=500,        # Ağaç sayısı
#     learning_rate=0.05,      # Öğrenme hızı (Küçük değerler daha iyidir)
#     max_depth=7,             # Ağaç derinliği
#     random_state=42, 
#     n_jobs=-1                # Tüm çekirdekleri kullan
# )
# model_xgb.fit(X_train, y_train)
# print("✅ Eğitim tamamlandı!")

# # 7️⃣ Tahmin ve performans ölçümü
# y_pred_xgb = model_xgb.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred_xgb)
# rmse = np.sqrt(np.mean((y_test - y_pred_xgb)**2))
# r2 = r2_score(y_test, y_pred_xgb)

# print("\n📊 XGBoost Model Performansı:")
# print(f"MAE  : {mae:.2f}")
# print(f"RMSE : {rmse:.2f}")
# print(f"R²   : {r2:.3f}")

# # 8️⃣ Modeli kaydet
# joblib.dump(model_xgb, "signal_strength_model_xgb.pkl")
# joblib.dump(le_network, "network_encoder.pkl") # Encoder'lar aynı kalır
# joblib.dump(le_timeofday, "timeofday_encoder.pkl")

# print("\n💾 XGBoost Modeli kaydedildi: signal_strength_model_xgb.pkl")

# =========================================
# 🎯 train_xgboost.py
# Amaç: processed_data.csv verisiyle XGBoost modelini eğitmek
# =========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor 
import joblib
import numpy as np

# 1️⃣ Veri Yükleme
data = pd.read_csv("../data/processed_data.csv")

# 2️⃣ Hedef ve özellikleri belirle
target = "Signal Strength (dBm)"
features = [
    "Latitude",
    "Longitude",
    "Signal Quality (%)",
    "Data Throughput (Mbps)",
    "Latency (ms)",
    "Hour",
    "DayOfWeek",
    "IsWeekend",
    "Network Type", 
    "BB60C Measurement (dBm)",
    "srsRAN Measurement (dBm)",
    "BladeRFxA9 Measurement (dBm)",
    "TimeOfDay"
]

# 3️⃣ Kategorik Özellikleri encode et (Aynı Encoder'lar)
le_network = LabelEncoder()
data["Network Type"] = le_network.fit_transform(data["Network Type"])

le_timeofday = LabelEncoder()
data["TimeOfDay"] = le_timeofday.fit_transform(data["TimeOfDay"])


# 4️⃣ Girdi ve hedef ayrımı
X = data[features]
y = data[target]

# 5️⃣ Eğitim / test seti bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Model oluşturma ve eğitme
print("🚀 XGBoost model training...")
# Temel XGBoost parametreleri
model_xgb = XGBRegressor(
    n_estimators=500,        # Ağaç sayısı
    learning_rate=0.05,      # Öğrenme hızı (Küçük değerler daha iyidir)
    max_depth=7,             # Ağaç derinliği
    random_state=42, 
    n_jobs=-1                # Tüm çekirdekleri kullan
)
model_xgb.fit(X_train, y_train)
print("✅ Training Completed!")

# 7️⃣ Tahmin ve performans ölçümü
y_pred_xgb = model_xgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_xgb)
rmse = np.sqrt(np.mean((y_test - y_pred_xgb)**2))
r2 = r2_score(y_test, y_pred_xgb)

print("\n📊 XGBoost Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# =========================================================
# 📝 YENİ ADIM: Analiz Dosyalarını Kaydetme (Aynı Dizine)
# =========================================================

# A. Özellik Önemini (Feature Importance) Kaydetme
print("\n📝 Özellik Önemini Kaydediyor...")

# 1. Modelden önem skorlarını ve özellik isimlerini al
# XGBoost'ta importance skorları model.feature_importances_ ile alınır
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model_xgb.feature_importances_
})

# 2. Önem sırasına göre sırala
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# 3. CSV dosyasına kaydet
importance_filename = "xgboost_feature_importance.csv" # Aynı dizine kaydet
feature_importances.to_csv(importance_filename, index=False)

print(f"✅ Özellik Önem Sırası Kaydedildi: {importance_filename}")


# B. Tahmin Sonuçlarını Gerçek Değerlerle Birlikte Kaydetme
print("📝 Tahmin Sonuçlarını Kaydediyor...")

# 1. Test setindeki gerçek X, gerçek y ve tahminleri tek bir DataFrame'de birleştir
# Not: y_test, Panda Serisi olduğu için .values ile NumPy array'e dönüştürülür
results_df = X_test.copy()
results_df['Signal Strength (Actual)'] = y_test.values
results_df['Signal Strength (Prediction)'] = y_pred_xgb

# 2. Tahmin Sonuçlarını CSV dosyasına kaydet
results_filename = "xgboost_prediction_results.csv" # Aynı dizine kaydet
results_df.to_csv(results_filename, index=False)

print(f"✅ Tahmin Sonuçları Kaydedildi: {results_filename}")


# 8️⃣ Modeli kaydet
joblib.dump(model_xgb, "signal_strength_model_xgb.pkl")
joblib.dump(le_network, "network_encoder.pkl") # Encoder'lar aynı kalır
joblib.dump(le_timeofday, "timeofday_encoder.pkl")

print("\n💾 XGBoost Modeli kaydedildi: signal_strength_model_xgb.pkl")