import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Analisis Clustering & Regresi COVID-19 Indonesia",
    layout="wide"
)

st.title("📊 Analisis Clustering & Regresi COVID-19 Indonesia")
st.write("Aplikasi ini menampilkan hasil **clustering provinsi** dan **regresi linear** COVID-19.")

# ===============================
# LOAD DATA (ABSOLUTE PATH)
# ===============================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "Covid-19_Indonesia_Dataset.csv")

    if not os.path.exists(data_path):
        st.error("❌ Dataset tidak ditemukan. Pastikan file ada di folder data/")
        st.stop()

    df = pd.read_csv(data_path)
    return df

df = load_data()

# ===============================
# PREPROCESSING
# ===============================
df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%m/%d/%Y')

df_cluster = df[df['Tanggal'] == '2022-09-15'].copy()
df_cluster = df_cluster[df_cluster['Provinsi'].notnull()]

df_cluster = df_cluster[
    [
        'Provinsi',
        'Total_Kasus',
        'Total_Kematian',
        'Total_Sembuh',
        'Kepadatan_Penduduk',
        'Populasi',
        'Total_Kasus_Per_Juta',
        'Total_Kematian_Per_Juta'
    ]
].copy()

df_cluster['Rasio_Kematian'] = df_cluster['Total_Kematian'] / df_cluster['Total_Kasus']
df_cluster['Rasio_Kesembuhan'] = df_cluster['Total_Sembuh'] / df_cluster['Total_Kasus']
df_cluster = df_cluster.dropna()

# ===============================
# CLUSTERING K-MEANS
# ===============================
features = df_cluster.drop(columns=['Provinsi'])

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(scaled_features)

# ===============================
# TAMPILKAN DATA CLUSTER
# ===============================
st.subheader("📌 Hasil Clustering Provinsi")
st.dataframe(df_cluster[['Provinsi', 'Cluster']])

# ===============================
# VISUALISASI CLUSTER
# ===============================
st.subheader("📈 Visualisasi Clustering")

fig1, ax1 = plt.subplots()
scatter = ax1.scatter(
    df_cluster['Total_Kasus_Per_Juta'],
    df_cluster['Total_Kematian_Per_Juta'],
    c=df_cluster['Cluster']
)
ax1.set_xlabel("Total Kasus per Juta")
ax1.set_ylabel("Total Kematian per Juta")
ax1.set_title("Clustering Provinsi COVID-19")

st.pyplot(fig1)

# ===============================
# REGRESI LINEAR
# ===============================
st.subheader("📉 Regresi Linear")

X = df_cluster[['Total_Kasus_Per_Juta']]
y = df_cluster['Total_Kematian_Per_Juta']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ===============================
# HASIL REGRESI
# ===============================
st.markdown("### ✅ Hasil Regresi Linear")
st.write(f"**R² Score :** {r2:.3f}")
st.write(f"**RMSE :** {rmse:.3f}")

# ===============================
# VISUALISASI REGRESI
# ===============================
fig2, ax2 = plt.subplots()
ax2.scatter(X_test, y_test)
ax2.plot(X_test, y_pred)
ax2.set_xlabel("Total Kasus per Juta")
ax2.set_ylabel("Total Kematian per Juta")
ax2.set_title("Regresi Linear COVID-19")

st.pyplot(fig2)

# ===============================
# KESIMPULAN
# ===============================
st.subheader("📝 Kesimpulan")
st.write("""
- Clustering membagi provinsi menjadi **3 kelompok** berdasarkan tingkat kasus dan kematian COVID-19.
- Regresi linear menunjukkan hubungan yang **kuat** antara jumlah kasus dan kematian.
- Nilai **R² mendekati 1** menandakan model cukup baik.
""")
