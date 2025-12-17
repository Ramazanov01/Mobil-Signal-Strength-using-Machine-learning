import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed_data.csv")

# Zaman - Sinyal gücü
plt.figure(figsize=(12,6))
sns.lineplot(x='Hour', y='Signal Strength (dBm)', data=df)
plt.title("Average Signal Strength by Hour of Day")
plt.show()

# Network tipi - Ortalama sinyal gücü
plt.figure(figsize=(8,5))
sns.barplot(x='Network Type', y='Signal Strength (dBm)', data=df)
plt.title("Signal Strength by Network Type")
plt.xticks(rotation=45)
plt.show()

# Lokasyon bazlı dağılım
plt.figure(figsize=(10,8))
sns.scatterplot(x='Longitude', y='Latitude', hue='Signal Strength (dBm)', data=df, palette='coolwarm')
plt.title("Signal Strength by Location")
plt.show()
 