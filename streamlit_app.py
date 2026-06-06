import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# ====================================================================
# PENTING: Ganti angka di bawah ini sesuai dengan ukuran input model Anda!
# Contoh: 150 jika model Anda pakai 150x150, atau 50 jika 50x50.
UKURAN_MODEL = 150  
# ====================================================================

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Deteksi Retak Beton (Crack Detection)",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Fungsi untuk Mengunduh Model dari Google Drive (Cached)
@st.cache_resource
def load_model_from_drive():
    file_id = '1s9SDPAdgWs2KkFipQ-tJL-zzP-DLIhFm'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model_crack_beton.h5'
    
    if not os.path.exists(output):
        with st.spinner("Sedang mengunduh model dari Google Drive... Harap tunggu (ini hanya dilakukan sekali)."):
            gdown.download(url, output, quiet=False)
            
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
def preprocess_image(image, target_size=(UKURAN_MODEL, UKURAN_MODEL)):
    # Pastikan format gambar adalah RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Ubah ukuran gambar sesuai kebutuhan model Anda
    image = image.resize(target_size)
    
    # Mengubah gambar menjadi array dan menormalisasi (skala 0-1)
    image_array = np.array(image) / 255.0
    
    # Menambahkan dimensi batch
    image_array = np.expand_dims(image_array, axis=0)
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
    
    # Memproses gambar dengan ukuran yang benar
    processed_image = preprocess_image(image, target_size=(UKURAN_MODEL, UKURAN_MODEL))
    
    # Melakukan Prediksi
    try:
        predictions = model.predict(processed_image)
        
        # 5. Logika Output Hasil Prediksi
        if predictions.shape[1] == 1:
            score = predictions[0][0]
            if score > 0.5:
                st.error(f"⚠️ **Hasil: RETAK** (Probabilitas: {score*100:.2f}%)")
            else:
                st.success(f"✅ **Hasil: TAK RETAK** (Probabilitas: {(1-score)*100:.2f}%)")
        else:
            # ====================================================================
            # BAGIAN YANG DIUBAH: Urutan kelas dibalik menjadi Tak Retak terlebih dahulu
            # ====================================================================
            class_names = ['Tak Retak', 'Retak'] 
            
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            hasil = class_names[predicted_class_idx]
            
            if hasil == 'Retak':
                st.error(f"⚠️ **Hasil: RETAK** (Tingkat Keyakinan: {confidence*100:.2f}%)")
            else:
                st.success(f"✅ **Hasil: TAK RETAK** (Tingkat Keyakinan: {confidence*100:.2f}%)")
                
    except Exception as prediction_error:
        st.error(f"Terjadi kesalahan saat prediksi: {prediction_error}")
        st.info(f"Coba periksa kembali parameter UKURAN_MODEL. Saat ini disetel ke: {UKURAN_MODEL}x{UKURAN_MODEL}")
