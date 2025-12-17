# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# import numpy as np # RMSE için lazım olabilir (eski scikit-learn sürümünde)

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
#     "TimeOfDay" # Yeni kategorik özellik olarak ekleniyor
# ]

# # 3️⃣ Kategorik Özellikleri encode et (Network Type ve TimeOfDay)
# le_network = LabelEncoder()
# data["Network Type"] = le_network.fit_transform(data["Network Type"])

# le_timeofday = LabelEncoder()
# # TimeOfDay'i encode et (örn: Morning, Afternoon, Evening, Night → 0,1,2,3)
# data["TimeOfDay"] = le_timeofday.fit_transform(data["TimeOfDay"])


# # 4️⃣ Girdi ve hedef ayrımı
# X = data[features]
# y = data[target]

# # 5️⃣ Eğitim / test seti bölünmesi
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# # 6️⃣ Model oluşturma ve eğitme
# print("🚀 Random Forest modeli eğitiliyor...")
# # n_estimators'ı artırmak performansı biraz daha artırabilir
# model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1) 
# model.fit(X_train, y_train)
# print("✅ Eğitim tamamlandı!")

# # 7️⃣ Tahmin ve performans ölçümü
# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)  # Calculate MSE first
# rmse = mse ** 0.5  # Then take square root to get RMSE
# r2 = r2_score(y_test, y_pred)

# print("\n📊 Model Performansı:")
# print(f"MAE  : {mae:.2f}")
# print(f"RMSE : {rmse:.2f}")
# print(f"R²   : {r2:.3f}")

# # 8️⃣ Modeli ve Encoder'ları kaydet
# joblib.dump(model, "signal_strength_model.pkl")
# # Birden fazla encoder'ı kaydetmek için
# joblib.dump(le_network, "network_encoder.pkl")
# joblib.dump(le_timeofday, "timeofday_encoder.pkl")


# print("\n💾 Model kaydedildi: signal_strength_model.pkl")
# print("💾 Encoder kaydedildi: network_encoder.pkl, timeofday_encoder.pkl")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np # RMSE için lazım olabilir (eski scikit-learn sürümünde)

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
    "TimeOfDay" # Yeni kategorik özellik olarak ekleniyor
]

# 3️⃣ Kategorik Özellikleri encode et (Network Type ve TimeOfDay)
le_network = LabelEncoder()
data["Network Type"] = le_network.fit_transform(data["Network Type"])

le_timeofday = LabelEncoder()
# TimeOfDay'i encode et (örn: Morning, Afternoon, Evening, Night → 0,1,2,3)
data["TimeOfDay"] = le_timeofday.fit_transform(data["TimeOfDay"])


# 4️⃣ Girdi ve hedef ayrımı
X = data[features]
y = data[target]

# 5️⃣ Eğitim / test seti bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 6️⃣ Model oluşturma ve eğitme
print("🚀 Random Forest model training...")
model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1) 
model.fit(X_train, y_train)
print("✅ Training Completed!")

# 7️⃣ Tahmin ve performans ölçümü
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)  # Calculate MSE first
rmse = mse ** 0.5  # Then take square root to get RMSE
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# =========================================================
# 📝 YENİ ADIM: Analiz Dosyalarını Kaydetme (Aynı Dizine)
# =========================================================

# A. Özellik Önemini (Feature Importance) Kaydetme
print("\n📝 Özellik Önemini Kaydediyor...")

# 1. Modelden önem skorlarını ve özellik isimlerini al (Random Forest için feature_importances_)
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
})

# 2. Önem sırasına göre sırala
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# 3. CSV dosyasına kaydet
importance_filename = "rf_feature_importance.csv" # rf (Random Forest) etiketi eklendi
feature_importances.to_csv(importance_filename, index=False)

print(f"✅ Özellik Önem Sırası Kaydedildi: {importance_filename}")


# B. Tahmin Sonuçlarını Gerçek Değerlerle Birlikte Kaydetme
print("📝 Tahmin Sonuçlarını Kaydediyor...")

# 1. Test setindeki gerçek X, gerçek y ve tahminleri tek bir DataFrame'de birleştir
results_df = X_test.copy()
results_df['Signal Strength (Actual)'] = y_test.values
results_df['Signal Strength (Prediction)'] = y_pred

# 2. Tahmin Sonuçlarını CSV dosyasına kaydet
results_filename = "rf_prediction_results.csv" # rf (Random Forest) etiketi eklendi
results_df.to_csv(results_filename, index=False)

print(f"✅ Tahmin Sonuçları Kaydedildi: {results_filename}")


# 8️⃣ Modeli ve Encoder'ları kaydet
joblib.dump(model, "signal_strength_model.pkl")
# Birden fazla encoder'ı kaydetmek için
joblib.dump(le_network, "network_encoder.pkl")
joblib.dump(le_timeofday, "timeofday_encoder.pkl")


print("\n💾 Model kaydedildi: signal_strength_model.pkl")
print("💾 Encoder kaydedildi: network_encoder.pkl, timeofday_encoder.pkl")