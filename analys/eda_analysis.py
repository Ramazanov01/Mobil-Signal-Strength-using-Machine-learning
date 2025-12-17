import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========== 1. Veri Yükleme ==========
data_path = os.path.join("data", "cleaned_data.csv")
df = pd.read_csv(data_path)

print("✅ Veri başarıyla yüklendi!")
print(df.head())

# ========== 2. Genel Bilgi ==========
print("\n📊 Veri Seti Bilgisi:")
print(df.info())

print("\n📈 Temel İstatistikler:")
print(df.describe())

# ========== 3. Sinyal Gücü Dağılımı ==========
signal_cols = [
    "Signal Strength (dBm)",
    "BB60C Measurement (dBm)",
    "srsRAN Measurement (dBm)",
    "BladeRFxA9 Measurement (dBm)"
]

plt.figure(figsize=(12, 6))
for col in signal_cols:
    if col in df.columns:
        sns.kdeplot(df[col], label=col)
plt.title("Sinyal Gücü Dağılımı (dBm)")
plt.xlabel("dBm")
plt.ylabel("Yoğunluk")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 4. Network Tipine Göre Ortalama Sinyal ==========
if "Network Type" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Network Type", y="Signal Strength (dBm)", data=df, estimator="mean", ci=None)
    plt.title("Network Tipine Göre Ortalama Sinyal Gücü")
    plt.xlabel("Network Tipi")
    plt.ylabel("Ortalama dBm")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

# ========== 5. Korelasyon Analizi ==========
plt.figure(figsize=(8, 6))
sns.heatmap(df[signal_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Sinyal Ölçümleri Arasındaki Korelasyon")
plt.tight_layout()
plt.show()

# ========== 6. Zaman Bazlı İnceleme ==========
if "Hour" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="Hour", y="Signal Strength (dBm)", data=df)
    plt.title("Saat Bazında Ortalama Sinyal Gücü")
    plt.xlabel("Saat")
    plt.ylabel("dBm")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\n✅ EDA tamamlandı! Grafikler gösterildi.")
