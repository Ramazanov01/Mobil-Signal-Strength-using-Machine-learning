# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import folium
# from folium.plugins import HeatMap
# import numpy as np

# # Görselleştirme için renk ve stil ayarları
# sns.set_style("whitegrid")
# plt.rcParams['figure.figsize'] = (10, 6)

# # Tahmin sonuç dosyanızı yükleyin (Örneğin LightGBM'in kaydettiği dosya)
# try:
#     results_df = pd.read_csv("../lightGbm_model/lgbm_prediction_results.csv")
# except FileNotFoundError:
#     print("HATA: 'lgbm_prediction_results.csv' dosyası bulunamadı. Lütfen dosya adını kontrol edin.")
#     exit()

# # Sütun isimlerini kısaltma
# ACTUAL = 'Signal Strength (Actual)'
# PREDICTION = 'Signal Strength (Prediction)'

# # =========================================================
# # 1. TEMEL İSTATİSTİKSEL GRAFİKLER (Matplotlib & Seaborn)
# # =========================================================

# print("📊 İstatistiksel Grafikler Hazırlanıyor...")

# # --- A. Sinyal Gücü Dağılımı (Histogram) ---
# plt.figure(figsize=(12, 5))
# sns.histplot(results_df[ACTUAL], kde=True, color="blue", label="Gerçek Değerler")
# plt.title('Sinyal Gücü Dağılımı (dBm)')
# plt.xlabel('Signal Strength (dBm)')
# plt.legend()
# plt.savefig('signal_distribution_histogram.png')
# plt.close()
# print("✅ Sinyal Dağılım Grafiği Kaydedildi: signal_distribution_histogram.png")

# # --- B. Gerçek vs Tahmin Edilen Değerler (Serpilme Grafiği) ---
# plt.figure(figsize=(8, 8))
# sns.scatterplot(x=results_df[ACTUAL], y=results_df[PREDICTION], alpha=0.6)

# # Mükemmel uyumu temsil eden 45 derecelik çizgi (y=x)
# min_val = results_df[[ACTUAL, PREDICTION]].min().min() - 5
# max_val = results_df[[ACTUAL, PREDICTION]].max().max() + 5
# plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='İdeal Uyum (Y=X)')

# plt.title('Gerçek Sinyal Gücü vs. Model Tahmini')
# plt.xlabel('Gerçek Signal Strength (dBm)')
# plt.ylabel('Tahmin Edilen Signal Strength (dBm)')
# plt.legend()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig('actual_vs_prediction_scatter.png')
# plt.close()
# print("✅ Gerçek vs Tahmin Grafiği Kaydedildi: actual_vs_prediction_scatter.png")


# # --- C. Hata Dağılımı (Residuals) ---
# results_df['Error'] = results_df[ACTUAL] - results_df[PREDICTION]
# plt.figure(figsize=(10, 6))
# sns.histplot(results_df['Error'], bins=50, kde=True)
# plt.title('Model Hata Dağılımı (Gerçek - Tahmin)')
# plt.xlabel('Hata (dBm)')
# plt.savefig('error_distribution_histogram.png')
# plt.close()
# print("✅ Hata Dağılım Grafiği Kaydedildi: error_distribution_histogram.png")


# # =========================================================
# # 2. COĞRAFİ GÖRSELLEŞTİRME (Folium Heatmap)
# # =========================================================

# print("\n🌍 Coğrafi Harita Hazırlanıyor (Bu biraz zaman alabilir)...")

# # 1. Haritanın merkezini belirleme (Verinin ortalama Lat/Lon değerleri)
# center_lat = results_df['Latitude'].mean()
# center_lon = results_df['Longitude'].mean()

# # 2. Haritayı oluşturma
# m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# # 3. Heatmap Verisini Hazırlama (Latitude, Longitude, Signal Strength)
# # Sadece Gerçek Sinyal Gücü Heatmap'i çiziyoruz.
# heat_data = [[row['Latitude'], row['Longitude'], row[ACTUAL]] for index, row in results_df.iterrows()]

# # 4. HeatMap'i haritaya ekleme
# HeatMap(heat_data, radius=15).add_to(m)

# # 5. Haritayı HTML dosyası olarak kaydetme
# map_filename = 'signal_strength_heatmap.html'
# m.save(map_filename)
# print(f"✅ Sinyal Yoğunluğu Haritası Kaydedildi: {map_filename} (HTML dosyasını tarayıcınızda açın)")

# print("\n✨ Tüm görselleştirmeler tamamlandı.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import numpy as np

# Visualization style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load the prediction results file
try:
    # NOTE: Assuming the path is correct for your system
    results_df = pd.read_csv("../lightGbm_model/lgbm_prediction_results.csv")
except FileNotFoundError:
    print("ERROR: 'lgbm_prediction_results.csv' file not found. Please check the file path.")
    exit()

# Column aliases
ACTUAL = 'Signal Strength (Actual)'
PREDICTION = 'Signal Strength (Prediction)'

# =========================================================
# 1. CORE STATISTICAL PLOTS (Matplotlib & Seaborn)
# =========================================================

print("📊 Preparing Statistical Plots...")

# --- A. Signal Strength Distribution (Histogram) ---
plt.figure(figsize=(12, 5))
# Title: Signal Strength Distribution (dBm)
# X-label: Signal Strength (dBm)
# Label: Actual Values
sns.histplot(results_df[ACTUAL], kde=True, color="blue", label="Actual Values")
plt.title('Signal Strength Distribution (dBm)')
plt.xlabel('Signal Strength (dBm)')
plt.legend()
plt.savefig('signal_distribution_histogram_en.png')
plt.close()
print("✅ Signal Distribution Histogram Saved: signal_distribution_histogram_en.png")

# --- B. Actual vs Predicted Values (Scatter Plot) ---
plt.figure(figsize=(8, 8))
# Title: Actual Signal Strength vs. Model Prediction
# X-label: Actual Signal Strength (dBm)
# Y-label: Predicted Signal Strength (dBm)
sns.scatterplot(x=results_df[ACTUAL], y=results_df[PREDICTION], alpha=0.6)

# Line representing perfect fit (Y=X)
min_val = results_df[[ACTUAL, PREDICTION]].min().min() - 5
max_val = results_df[[ACTUAL, PREDICTION]].max().max() + 5
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Fit (Y=X)')

plt.title('Actual Signal Strength vs. Model Prediction')
plt.xlabel('Actual Signal Strength (dBm)')
plt.ylabel('Predicted Signal Strength (dBm)')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('actual_vs_prediction_scatter_en.png')
plt.close()
print("✅ Actual vs Predicted Scatter Plot Saved: actual_vs_prediction_scatter_en.png")


# --- C. Error Distribution (Residuals) ---
results_df['Error'] = results_df[ACTUAL] - results_df[PREDICTION]
plt.figure(figsize=(10, 6))
# Title: Model Error Distribution (Actual - Predicted)
# X-label: Error (dBm)
sns.histplot(results_df['Error'], bins=50, kde=True)
plt.title('Model Error Distribution (Actual - Predicted)')
plt.xlabel('Error (dBm)')
plt.savefig('error_distribution_histogram_en.png')
plt.close()
print("✅ Error Distribution Histogram Saved: error_distribution_histogram_en.png")


# =========================================================
# 2. GEOSPATIAL VISUALIZATION (Folium Heatmap)
# =========================================================

print("\n🌍 Preparing Geospatial Heatmap (This may take a moment)...")

# 1. Determine map center (mean Lat/Lon)
center_lat = results_df['Latitude'].mean()
center_lon = results_df['Longitude'].mean()

# 2. Create the map
m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron") # Using a cleaner tile set

# 3. Prepare Heatmap Data (Actual Signal Strength)
heat_data = [[row['Latitude'], row['Longitude'], row[ACTUAL]] for index, row in results_df.iterrows()]

# 4. Add HeatMap to the map
HeatMap(heat_data, radius=15).add_to(m)

# 5. Save the map as an HTML file
map_filename = 'signal_strength_heatmap_en.html'
m.save(map_filename)
print(f"✅ Signal Strength Heatmap Saved: {map_filename} (Open the HTML file in your browser)")

print("\n✨ All visualizations completed and saved with English labels.")