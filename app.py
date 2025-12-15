import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Analisis COVID-19 Indonesia",
    layout="wide"
)

st.title("📊 Analisis Clustering & Regresi COVID-19 Indonesia")
st.write("Aplikasi ini menampilkan clustering provinsi dan regresi linear COVID-19.")

# ==================================================
# LOAD DATA
# ==================================================
st.sidebar.header("⚙️ Pengaturan")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    return df

df = load_data("data/Covid-19_Indonesia_Dataset.csv")

tanggal = st.sidebar.date_input(
    "Pilih Tanggal",
    pd.to_datetime("2022-09-15")
)

df_cluster = df[df['Tanggal'] == pd.to_datetime(tanggal)].copy()
df_cluster = df_cluster[df_cluster['Provinsi'].notnull()]

df_cluster = df_cluster[
    ['Provinsi','Total_Kasus','Total_Kematian','Total_Sembuh',
     'Kepadatan_Penduduk','Populasi',
     'Total_Kasus_Per_Juta','Total_Kematian_Per_Juta']
]

df_cluster = df_cluster[df_cluster['Total_Kasus'] > 0]

# Rasio
df_cluster['Rasio_Kematian'] = df_cluster['Total_Kematian'] / df_cluster['Total_Kasus']
df_cluster['Rasio_Kesembuhan'] = df_cluster['Total_Sembuh'] / df_cluster['Total_Kasus']

st.subheader("📂 Data COVID-19 (Preview)")
st.dataframe(df_cluster.head())

# ==================================================
# CLUSTERING
# ==================================================
st.subheader("🔗 K-Means Clustering Provinsi")

fitur = [
    'Total_Kasus','Total_Kematian','Total_Sembuh',
    'Populasi','Total_Kasus_Per_Juta',
    'Total_Kematian_Per_Juta',
    'Kepadatan_Penduduk',
    'Rasio_Kematian','Rasio_Kesembuhan'
]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_cluster[fitur])

# Elbow Method
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(scaled_data)
    wcss.append(km.inertia_)

fig_elbow, ax = plt.subplots()
ax.plot(range(1,11), wcss, marker='o')
ax.set_title("Elbow Method")
ax.set_xlabel("Jumlah Cluster")
ax.set_ylabel("WCSS")
st.pyplot(fig_elbow)

# Pilih jumlah cluster
k = st.slider("Pilih Jumlah Cluster", 2, 6, 5)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualisasi cluster
fig_cluster, ax2 = plt.subplots(figsize=(7,5))
sns.scatterplot(
    data=df_cluster,
    x='Kepadatan_Penduduk',
    y='Total_Kematian_Per_Juta',
    hue='Cluster',
    palette='viridis',
    s=100,
    ax=ax2
)
ax2.set_title("Sebaran Cluster Provinsi")
ax2.set_xlabel("Kepadatan Penduduk")
ax2.set_ylabel("Total Kematian per Juta")
st.pyplot(fig_cluster)

# ==================================================
# REGRESI LINEAR
# ==================================================
st.subheader("📈 Regresi Linear")

X = df_cluster[
    ['Total_Kasus_Per_Juta','Kepadatan_Penduduk','Rasio_Kematian']
]
y = df_cluster['Total_Kematian_Per_Juta']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

col1, col2 = st.columns(2)
col1.metric("R² Score", round(r2, 3))
col2.metric("RMSE", round(rmse, 2))

# Koefisien regresi
coef_df = pd.DataFrame({
    "Variabel": X.columns,
    "Koefisien": model.coef_
})

st.write("📌 Koefisien Regresi")
st.dataframe(coef_df)

# Visualisasi aktual vs prediksi
fig_reg, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred)
ax3.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)
ax3.set_xlabel("Nilai Aktual")
ax3.set_ylabel("Nilai Prediksi")
ax3.set_title("Regresi Linear: Aktual vs Prediksi")
st.pyplot(fig_reg)

# ==================================================
# KESIMPULAN
# ==================================================
st.subheader("🧠 Kesimpulan")
st.write("""
- K-Means Clustering digunakan untuk mengelompokkan provinsi berdasarkan karakteristik COVID-19.
- Regresi linear digunakan untuk memprediksi jumlah kematian per juta penduduk.
- Nilai R² yang tinggi menunjukkan model cukup baik dalam menjelaskan data.
""")
