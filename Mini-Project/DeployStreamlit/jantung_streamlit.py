import streamlit as st
import joblib
import numpy as np
# cara jalanka: streamlit run jantung_streamlit.py

# Menggunakan st.cache_resource untuk memuat pipeline hanya sekali
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load('pipeline_jantung.joblib')
        return pipeline
    except FileNotFoundError:
        st.error("File 'pipeline_jantung.joblib' tidak ditemukan. Pastikan sudah disimpan!")
        return None

loaded_pipeline = load_pipeline()

st.set_page_config(page_title="Prediksi Penyakit Jantung")

st.title("Aplikasi Prediksi Penyakit Jantung")
st.markdown("Isi data berikut untuk mendapatkan prediksi risiko penyakit jantung:")

with st.form(key='my_prediction_form'):
    st.sidebar.header("Input Data Pasien")

    # --- Membuat Input untuk 13 Fitur Tanpa Nilai Default ---
    age = st.sidebar.number_input("Umur (age)", min_value=0, max_value=120)
    sex = st.sidebar.selectbox("Jenis Kelamin (sex)", options=[None, 1, 0], format_func=lambda x: "Pilih" if x is None else ("Pria" if x == 1 else "Wanita"))
    cp = st.sidebar.selectbox("Tipe Nyeri Dada (cp)", options=[None, 0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Tekanan Darah Istirahat (trestbps)", min_value=50, max_value=200)
    chol = st.sidebar.number_input("Kolesterol (chol)", min_value=100, max_value=600)
    fbs = st.sidebar.selectbox("Gula Darah Puasa > 120 mg/dl (fbs)", options=[None, 1, 0], format_func=lambda x: "Pilih" if x is None else ("Ya" if x == 1 else "Tidak"))
    restecg = st.sidebar.selectbox("Hasil EKG Istirahat (restecg)", options=[None, 0, 1, 2])
    thalach = st.sidebar.number_input("Detak Jantung Maks. (thalach)", min_value=60, max_value=220)
    exang = st.sidebar.selectbox("Angina Akibat Latihan (exang)", options=[None, 1, 0], format_func=lambda x: "Pilih" if x is None else ("Ya" if x == 1 else "Tidak"))
    oldpeak = st.sidebar.number_input("Oldpeak (depresi ST)", min_value=0.0, max_value=6.2, step=0.1)
    slope = st.sidebar.selectbox("Kemiringan Puncak ST (slope)", options=[None, 0, 1, 2])
    ca = st.sidebar.selectbox("Jumlah Pembuluh Darah Utama (ca)", options=[None, 0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalium Stress Test (thal)", options=[None, 0, 1, 2, 3])

    submit_button = st.form_submit_button(label='Prediksi')

if submit_button:
    # Cek apakah semua input sudah diisi
    input_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    if None in input_values:
        st.warning("Mohon lengkapi semua data.")
    else:
        # PENTING: Konversi ke float64 agar sesuai dengan StandardScaler
        features_array = np.array(input_values).astype(np.float64).reshape(1, -1)
        
        if loaded_pipeline is not None:
            prediction_result = loaded_pipeline.predict(features_array)
            
            st.markdown("---")
            if prediction_result[0] == 1:
                st.error("⚠️ **Hasil: Pasien memiliki kemungkinan penyakit jantung.**")
            else:
                st.success("✅ **Hasil: Pasien kemungkinan tidak memiliki penyakit jantung.**")
            
            st.info("Disclaimer: Hasil ini hanya berdasarkan model AI. Konsultasikan dengan profesional medis untuk diagnosis akurat.")
        else:
            st.warning("Model belum dimuat. Mohon cek file pipeline.")