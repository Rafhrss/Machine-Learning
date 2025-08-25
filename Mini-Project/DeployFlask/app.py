import pickle
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# with open('model_knn.joblib', 'rb') as model_file:
#     loaded_model_knn = pickle.load(model_file)

loaded_model_knn = joblib.load('model_knn.joblib')

@app.route('/')
def home():
    return render_template(template_name_or_list='home.html', title='Home')

@app.route('/prediction')
def prediction_form():
    return render_template('prediction.html', title='Prediction Home')

@app.post('/predict')
def prediction():
    # Mengambil data dari form
    data = [
        request.form.get('age'),
        request.form.get('sex'),
        request.form.get('cp'),
        request.form.get('trestbps'),
        request.form.get('chol'),
        request.form.get('fbs'),
        request.form.get('restecg'),
        request.form.get('thalach'),
        request.form.get('exang'),
        request.form.get('oldpeak'),
        request.form.get('slope'),
        request.form.get('ca'),
        request.form.get('thal')
    ]

    # Mengubah data menjadi array numpy dan tipe float
    final_features = np.array(data, dtype=np.float64).reshape(1, -1)

    # Melakukan prediksi menggunakan model
    prediction_result = loaded_model_knn.predict(final_features)
    
    # Mengubah hasil prediksi (0 atau 1) menjadi teks 'Tidak' atau 'Ya'
    if prediction_result[0] == 1:
        output = 'Ya, ada kemungkinan penyakit jantung.'
    else:
        output = 'Tidak, kemungkinan tidak ada penyakit jantung.'

    # Menampilkan hasil di halaman result.html
    return render_template('result.html', prediction_text=output, title='Hasil Prediction')




# flask --app=app run --debug --reload
# atau yg ini
# app.run(debug=True)
