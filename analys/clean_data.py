import pandas as pd

df = pd.read_csv("data/data.csv")

# 1. Gereksiz sütunları çıkar
df = df.drop(columns=["Timestamp", "Locality"])

# 2. 0 olan ölçümleri at (bunlar hatalı kayıt gibi duruyor)
df = df[(df["Signal Strength (dBm)"] < 0) & (df["BB60C Measurement (dBm)"] < 0)]

# 3. Network Type’ı kontrol et
print("🔹 Benzersiz Ağ Türleri:")
print(df["Network Type"].unique())

# 4. Temiz veriyi kaydet
df.to_csv("data/cleaned_data.csv", index=False)
print(f"✅ Temizlenmiş veri kaydedildi: {len(df)} satır kaldı.")
