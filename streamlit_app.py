import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Deteksi Retak Beton (Crack Detection)",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Fungsi untuk Mengunduh Model dari Google Drive (Cached)
@st.cache_resource
def load_model_from_drive():
    # ID File dari link Google Drive yang Anda berikan
    file_id = '1s9SDPAdgWs2KkFipQ-tJL-zzP-DLIhFm'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Tempat penyimpanan sementara di server/lokal
    output = 'model_crack_beton.h5'
    
    # Unduh file jika belum ada di direktori kerja
    if not os.path.exists(output):
        with st.spinner("Sedang mengunduh model dari Google Drive... Harap tunggu (ini hanya dilakukan sekali)."):
            gdown.download(url, output, quiet=False)
            
    # Load model menggunakan TensorFlow Keras
    model = tf.keras.models.load_model(output)
    return model

# Load model ke dalam aplikasi
try:
    model = load_model_from_drive()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# 3. Fungsi Preprocessing Gambar
def preprocess_image(image, target_size=(224, 224)):
    """
    Sesuaikan 'target_size' dengan ukuran input dari model AI Anda 
    (misal: 224x224, 150x150, atau 50x50).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalisasi jika model Anda melatih dengan skala 0-1
    image_array = np.expand_dims(image_array, axis=0) # Tambah dimensi batch
    return image_array

# 4. Antarmuka Pengguna (UI) Aplikasi
st.title("🛡️ Sistem Deteksi Retak Beton")
st.write("Unggah foto permukaan beton untuk melihat apakah terdapat keretakan.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    
    st.write("---")
    st.write("🔄 Sedang memproses klasifikasi...")
    
    # Preprocessing (Sesuaikan target_size dengan arsitektur model Anda)
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    # Melakukan Prediksi
    predictions = model.predict(processed_image)
    
    # 5. Logika Output Hasil Prediksi
    # Catatan: Sesuaikan logika di bawah ini dengan output layer model Anda (Binary atau Categorical)
    
    # JIKA MODEL ADALAH BINARY (Menggunakan aktivasi Sigmoid, output berupa 1 nilai probabilitas)
    if predictions.shape[1] == 1:
        score = predictions[0][0]
        # Misal: mendekati 1 berarti Retak, mendekati 0 berarti Tak Retak (atau sebaliknya)
        if score > 0.5:
            st.error(f"⚠️ **Hasil: RETAK** (Probabilitas: {score*100:.2f}%)")
        else:
            st.success(f"✅ **Hasil: TAK RETAK** (Probabilitas: {(1-score)*100:.2f}%)")
            
    # JIKA MODEL ADALAH CATEGORICAL (Menggunakan aktivasi Softmax, output berupa array kelas)
    else:
        class_names = ['Retak', 'Tak Retak'] # Urutkan sesuai index labeling pada saat training
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        hasil = class_names[predicted_class_idx]
        
        if hasil == 'Retak':
            st.error(f"⚠️ **Hasil: RETAK** (Tingkat Keyakinan: {confidence*100:.2f}%)")
        else:
            st.success(f"✅ **Hasil: TAK RETAK** (Tingkat Keyakinan: {confidence*100:.2f}%)")
