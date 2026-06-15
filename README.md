# Machine Learning Repository

Selamat datang di repository Machine Learning saya! Repository ini berisi berbagai pembelajaran, proyek, dan eksperimen terkait *Machine Learning*, mulai dari pemrosesan data hingga pengembangan model.

## Struktur Direktori

### `ML-Dicoding/`
Direktori ini berisi pekerjaan dan submission untuk kelas Machine Learning dari Dicoding. Proyek utama di dalam direktori ini adalah pengintegrasian metode *Unsupervised Learning* (Clustering) dan *Supervised Learning* (Klasifikasi) untuk memecahkan sebuah studi kasus nyata.

- **Studi Kasus**: Analisis pola aktivitas dan transaksi keuangan untuk *fraud detection* (deteksi penipuan) menggunakan modifikasi dari *Bank Transaction Dataset for Fraud Detection*.
- **Tujuan**:
  1. Melakukan *Clustering* untuk menghasilkan label/kelas (misalnya tingkat aktivitas/risiko) pada dataset transaksi yang belum berlabel.
  2. Melakukan *Klasifikasi* menggunakan dataset yang telah dilabeli oleh model clustering tersebut untuk memprediksi kelas pada data yang akan datang.
- **Tahapan Proyek**:
  - *Exploratory Data Analysis* (EDA)
  - Data Preprocessing (Penanganan Missing Values, Encoding, Scaling, dll)
  - Pemodelan Clustering (K-Means) & Evaluasi (Elbow Method & Silhouette Score)
  - Pemodelan Klasifikasi & Evaluasi Metrik

## Kebutuhan Sistem
Untuk menjalankan *notebook* di repository ini, disarankan untuk menginstal pustaka-pustaka berikut:
- Python 3.x
- Jupyter Notebook
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn` (direkomendasikan versi 1.7.0 atau terbaru)
- `yellowbrick`