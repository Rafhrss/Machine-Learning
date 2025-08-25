#cara jalankannya:   streamlit run visualisasi.py
# jalankan secara terpisah

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Visualisasi Data Penyakit Jantung", layout="wide")

st.title("Visualisasi Dataset Penyakit Jantung")
st.markdown("Menganalisis distribusi dan korelasi fitur-fitur dalam dataset.")

# --- Fungsi untuk memuat data dari file CSV asli ---
# Gunakan st.cache_data agar data hanya dimuat sekali
@st.cache_data
def load_data():
    try:
        # Ganti 'heart.csv' dengan nama file yang kamu gunakan
        # Pastikan file ini ada di direktori yang sama
        df = pd.read_csv('heart.csv')
        return df
    except FileNotFoundError:
        st.error("File 'heart.csv' tidak ditemukan. Mohon unggah file atau pastikan namanya benar.")
        return None

df = load_data()

if df is not None:
    # --- Visualisasi 1: Distribusi Diagnosis ---
    st.subheader("1. Distribusi Pasien (Sehat vs Sakit)")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='target', data=df, ax=ax1)
    ax1.set_title('Distribusi Diagnosis Penyakit Jantung')
    ax1.set_xlabel('Diagnosis (0: Sehat, 1: Sakit)')
    ax1.set_ylabel('Jumlah Pasien')
    st.pyplot(fig1)

    # --- Visualisasi 2: Heatmap Korelasi ---
    st.subheader("2. Matriks Korelasi Antar Fitur")
    st.write("Heatmap ini menunjukkan hubungan antara setiap fitur. Nilai mendekati +1 atau -1 menunjukkan korelasi kuat.")

    # Hitung matriks korelasi
    corr_matrix = df.corr()

    # Buat figure dan axes untuk plot
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    ax2.set_title('Matriks Korelasi Fitur')
    st.pyplot(fig2)