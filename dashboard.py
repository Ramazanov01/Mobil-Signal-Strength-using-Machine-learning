import streamlit as st
import requests
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static

# FastAPI API'nizin adresi
API_URL = "http://127.0.0.1:8000/predict_signal"

# 1. Sayfa Yapılandırması
st.set_page_config(layout="wide")
st.title("📡 Mobil Sinyal Gücü Tahmin Dashboard'u")
st.markdown("---")

# 2. Sidebar'da Giriş Formu Oluşturma
st.sidebar.header("Anlık Tahmin Girişi")

with st.sidebar.form("prediction_form"):
    st.markdown("**Konumsal & Ağ Bilgileri**")
    
    # Giriş Alanları
    lat = st.number_input('Latitude', value=25.60, format="%.4f")
    lon = st.number_input('Longitude', value=85.14, format="%.4f")
    
    network_type = st.selectbox('Network Type', ['3G', '4G', 'LTE', '5G'])
    
    # Kalan 10 özellik için örnek basit girdiler (API'nin beklediği tam liste)
    st.markdown("**Ölçüm & Performans**")
    sig_quality = st.number_input('Signal Quality (%)', value=90.0, min_value=0.0, max_value=100.0)
    throughput = st.number_input('Data Throughput (Mbps)', value=50.0)
    latency = st.number_input('Latency (ms)', value=40.0)
    
    bb60c = st.number_input('BB60C Measurement (dBm)', value=-70.0, format="%.1f")
    srsran = st.number_input('srsRAN Measurement (dBm)', value=-73.0, format="%.1f")
    bladerfx = st.number_input('BladeRFxA9 Measurement (dBm)', value=-71.0, format="%.1f")
    
    st.markdown("**Zaman Bilgisi**")
    hour = st.slider('Hour', 0, 23, 15)
    day_of_week = st.slider('DayOfWeek', 0, 6, 2)
    is_weekend = st.selectbox('IsWeekend', [0, 1])
    time_of_day = st.selectbox('TimeOfDay', ['Morning', 'Afternoon', 'Evening', 'Night'])
    
    submitted = st.form_submit_button("Sinyal Gücünü Tahmin Et")

# 3. Form Gönderimi ve API İsteği
if submitted:
    # 3.1. API'ye gönderilecek JSON verisini oluştur
    payload = {
        "Latitude": lat,
        "Longitude": lon,
        "Signal_Quality_percent": sig_quality,
        "Data_Throughput_Mbps": throughput,
        "Latency_ms": latency,
        "Hour": hour,
        "DayOfWeek": day_of_week,
        "IsWeekend": is_weekend,
        "Network_Type": network_type,
        "BB60C_Measurement_dBm": bb60c,
        "srsRAN_Measurement_dBm": srsran,
        "BladeRFxA9_Measurement_dBm": bladerfx,
        "TimeOfDay": time_of_day
    }
    
    # 3.2. API'ye POST isteği gönder
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Hata kodu gelirse istisna fırlat
        
        result = response.json()
        predicted_signal = result['predicted_signal_dBm']
        
        # 3.3. Sonucu gösterme
        st.sidebar.success(f"**Tahmin Edilen Sinyal:** {predicted_signal} dBm")
        st.balloons()
        
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"API bağlantı hatası: {e}. FastAPI sunucusunun çalıştığından emin olun.")


# 4. Ana Panelde Harita ve Analiz Alanı
st.header("Coğrafi Sinyal Yoğunluğu Haritası")

# Harita verisini yükleme (Önceki kodda kaydettiğiniz CSV dosyasından)
try:
    map_data = pd.read_csv("lightGbm_model/lgbm_prediction_results.csv")
except FileNotFoundError:
    st.error("Harita verisi (lgbm_prediction_results.csv) bulunamadı.")
    map_data = pd.DataFrame({'Latitude': [25.6], 'Longitude': [85.1]})


# 4.1. Folium Haritasını Oluşturma
center_lat = map_data['Latitude'].mean()
center_lon = map_data['Longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Heatmap Verisi (Gerçek Sinyal Gücü)
# Sütun adı kontrolü yapılmalıdır, buradaki tahmin/gerçek sütun adlarını kullanıyoruz
ACTUAL = 'Signal Strength (Actual)'
if ACTUAL in map_data.columns:
    heat_data = [[row['Latitude'], row['Longitude'], row[ACTUAL]] for index, row in map_data.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m)
    st.caption("Gösterilen harita, modelin eğitildiği verideki Gerçek Sinyal Gücü (Actual) dağılımıdır.")
else:
    st.warning("Heatmap oluşturulamadı: 'Signal Strength (Actual)' sütunu dosyada bulunamadı.")


# 4.2. Streamlit'e Haritayı Gösterme
folium_static(m, width=900, height=550)

st.markdown("---")
st.subheader("Model Performansı Özeti")
st.markdown("**R² Skoru:** 0.895 (Optimize edilmiş LightGBM Modelinden)")