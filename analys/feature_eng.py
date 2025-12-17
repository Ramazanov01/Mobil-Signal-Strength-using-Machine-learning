# feature_engineering.py

import pandas as pd

# 1️⃣ CSV dosyasını oku
df = pd.read_csv("data/data.csv")

# 2️⃣ Timestamp sütununu datetime formatına çevir
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# 3️⃣ Yeni zaman tabanlı feature’lar çıkar
df['Hour'] = df['Timestamp'].dt.hour                # Günün saati
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek      # 0 = Pazartesi, 6 = Pazar
df['DayName'] = df['Timestamp'].dt.day_name()       # Gün ismi (opsiyonel)
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # 1 = Cumartesi/Pazar

# 4️⃣ Günün saatine göre kategorik aralık (örnek)
def time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df['TimeOfDay'] = df['Hour'].apply(time_of_day)

# 5️⃣ Gereksiz Timestamp’ı koru ya da sil (sana kalmış)
# df = df.drop(columns=['Timestamp'])

# 6️⃣ Yeni veriyi kaydet
df.to_csv("data/processed_data.csv", index=False)

# 7️⃣ Kontrol için birkaç satır yazdır
print("✅ Yeni feature’lar eklendi!")
print(df[['Timestamp', 'Hour', 'DayOfWeek', 'IsWeekend', 'TimeOfDay']].head())
