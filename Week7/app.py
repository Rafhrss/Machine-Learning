import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# LEVEL 1

# st.title("Hello World")
# st.write("ini Aplikasi ke 10 aku menggunakan Streamlit.")


# nama = st.text_input('Masukan Nama: ')
# umur = st.number_input('masukan Umur: ', min_value=0)

# if st.button('klik ini'):
#     if umur < 18:
#         st.error(f'maaf {nama} kamu belum cukup umur')
#     else:
#         st.success('kamu sudah dewasa')




# LEVEL 2

# st.title("Level 2 EDA Sederhana")
# st.write("ini Aplikasi ke 10 aku menggunakan Streamlit.")

# file_upload = st.file_uploader("Upload File CSV", type='csv')

# if file_upload is not None:
#     df = pd.read_csv(file_upload)
#     st.write('Dataset berhasil dimuat!')
#     st.dataframe(df.head())
#     # Ringkasan data
#     if st.button('Buat ringkasan data'):
#         st.write(df.describe())

# if file_upload is not None:
#     st.subheader('visualisasi data')

#     num_cols = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
#     col = st.selectbox('Pilih kolom numerik: ', num_cols)
    
#     #histogram
#     fig, ax = plt.subplots()
#     df[col].hist(bins=20, ax=ax)
#     st.pyplot(fig)

#     st.line_chart(df[num_cols])



# Level 3

# model, target_names = joblib.load('iris_model.pkl')   #model, target class (satosa, virginica, versicolor)

# st.title('Prediksi Spesies Bunga Iris 🌸🌼🌼')
# st.markdown('''
# Aplikasi ini memprediksi spesies bunga **Iris** berdasarkan panjang & lebar sepal dan petal.
# ''')

# st.sidebar.header('Input Fitur Bunga')
# sepal_length = st.sidebar.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
# sepal_width = st.sidebar.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
# petal_length = st.sidebar.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
# petal_width = st.sidebar.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# # Level 4

# features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# if st.button("Prediksi"):
#     pred = model.predict(features)[0]
#     proba = model.predict_proba(features)[0]

#     st.subheader("Hasil Prediksi 🌼")
#     st.success(f"Spesies: {target_names[pred]}")
#     st.progress(int(proba[pred]*100))
#     st.write("Probabilitas detail:")
#     for cls, p in zip(target_names, proba):
#         st.write(f"- {cls}: {p:.2f}")

#     fig, ax = plt.subplots()
#     ax.bar(target_names, proba, color=["#FF9999","#66B2FF","#99FF99"])
#     ax.set_ylabel("Probabilitas")
#     st.pyplot(fig)

#     col1, col2 = st.columns(2)
#     col1.metric("Sepal Length", sepal_length)
#     col2.metric("Petal Length", petal_length)




# Level 5

# sebuah dekorator / alias untuk nge run sebuah function, ketika aliasnya sama dengan run maka akan jalan


# === Load Model (full model, bukan state_dict) ===
@st.cache_resource
def load_model():
    model = torch.load("cnn_mnist.pth", map_location="cpu", weights_only=False)
    model.eval()
    return model

model = load_model()

# === Streamlit UI ===
st.title("✏️ Prediksi Angka Tulis Tangan (MNIST CNN)")
st.markdown("Gambarlah angka 0-9 di kanvas, lalu klik **Prediksi**")

# === Canvas untuk menggambar ===
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,           # biar coretan lebih tebal
    stroke_color="white",
    background_color="black",
    width=280,                 # kanvas besar
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess(img):
    # resize & grayscale
    img = img.resize((28, 28)).convert("L")
    img = np.array(img).astype("float32") / 255.0

    # --- TES: jangan di-invert dulu ---
    # MNIST asli: digit putih (1), background hitam (0)
    # Jadi coretan putih di canvas seharusnya sudah sesuai
    # img = 1.0 - img   <-- coba matikan dulu

    # Tambahkan threshold biar lebih jelas (optional)
    img = (img > 0.5).astype("float32")

    # Bentuk tensor [1,1,28,28]
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img


# === Prediksi ===
if st.button("Prediksi"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"))
        input_tensor = preprocess(img)

        # === PREVIEW INPUT 28x28 ===
        small_img = input_tensor.squeeze().numpy()  # [28,28]
        st.image(
            Image.fromarray((small_img * 255).astype("uint8")).resize((140, 140)),
            caption="Input 28x28 ke Model",
            width=140
        )

        # === PREDIKSI ===
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            probs = torch.softmax(output, dim=1).numpy()[0]

        st.success(f"✅ Hasil Prediksi: {pred}")
        st.bar_chart(probs)
    else:
        st.warning("Silakan gambar angka dulu di kanvas!")

