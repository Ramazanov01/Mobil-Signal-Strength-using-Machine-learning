import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("data/data.csv")

# İlk 5 satırı göster
print("🔹 İlk 5 satır:")
print(df.head())

# Veri tipi ve eksik değer kontrolü
print("\n🔹 Veri Bilgisi:")
print(df.info())

print("\n🔹 Eksik değer sayısı:")
print(df.isnull().sum())

# İstatistiksel özet
print("\n🔹 İstatistiksel özet:")
print(df.describe())
