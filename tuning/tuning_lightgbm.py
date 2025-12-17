# =========================================
# 🔬 tuning_lightgbm.py
# Amaç: LightGBM için en iyi hiperparametreleri bulmak
# =========================================

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import numpy as np

# 1️⃣ Veri Yükleme ve Ön İşleme (Aynı Adımlar)
data = pd.read_csv("../data/processed_data.csv")
target = "Signal Strength (dBm)"
features = [
    # ... (tüm 13 özellik listesi) ...
    "Latitude", "Longitude", "Signal Quality (%)", "Data Throughput (Mbps)",
    "Latency (ms)", "Hour", "DayOfWeek", "IsWeekend", "Network Type", 
    "BB60C Measurement (dBm)", "srsRAN Measurement (dBm)", 
    "BladeRFxA9 Measurement (dBm)", "TimeOfDay"
]

le_network = LabelEncoder()
data["Network Type"] = le_network.fit_transform(data["Network Type"])
le_timeofday = LabelEncoder()
data["TimeOfDay"] = le_timeofday.fit_transform(data["TimeOfDay"])

X = data[features]
y = data[target]

# 5️⃣ Eğitim / test seti bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# 6️⃣ HİPERPARAMETRE AYARI
# =========================================================

print("🔎 For LightGBM Hiperparameter Starting...")

# Denenecek parametre aralıkları
param_grid = {
    'n_estimators': [300, 500, 700],  # Ağaç sayısını dene
    'max_depth': [5, 7, 10],          # Maksimum derinliği dene
    'learning_rate': [0.05, 0.03, 0.01], # Öğrenme hızını dene
}

# LGBM modelini varsayılan değerlerle başlat
lgbm = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

# GridSearchCV'yi kur (R² skorunu optimize etsin, 3 katmanlı çapraz doğrulama ile)
grid_search = GridSearchCV(
    estimator=lgbm, 
    param_grid=param_grid, 
    scoring='r2', 
    cv=3,                 
    verbose=2,
    n_jobs=-1
)

# Aramayı başlat (Bu işlem VEREBİLİR)
grid_search.fit(X_train, y_train)

print("\n✅  Tuning Completed!")
print("-" * 40)
print(f"🥇 Best R² Score: {grid_search.best_score_:.4f}")
print(f"⚙️ Best Parameters: {grid_search.best_params_}")
print("-" * 40)

# 7️⃣ Sonraki Adım: Bulunan bu parametreleri alıp train_lightgbm.py dosyasına yapıştırın.