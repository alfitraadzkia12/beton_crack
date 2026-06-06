import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =====================
# PAGE CONFIG
# =====================

st.set_page_config(
    page_title="Deteksi Retak Beton",
    page_icon="🏗️",
    layout="wide"
)

# =====================
# CSS
# =====================

st.markdown("""
<style>

.stApp{
background: linear-gradient(
135deg,
#0f172a,
#1e293b,
#0f172a
);
}

.main-title{
text-align:center;
font-size:48px;
font-weight:bold;
color:white;
}

.sub-title{
text-align:center;
color:#cbd5e1;
font-size:18px;
margin-bottom:30px;
}

.card{
background:#1e293b;
padding:20px;
border-radius:20px;
box-shadow:0px 0px 20px rgba(0,0,0,0.3);
}

.result-card{
background:#243447;
padding:20px;
border-radius:20px;
}

.pred-box{
padding:15px;
border-radius:12px;
font-size:22px;
font-weight:bold;
color:white;
}

.crack{
background:#b91c1c;
}

.normal{
background:#15803d;
}

</style>
""", unsafe_allow_html=True)

# =====================
# HEADER
# =====================

st.markdown("""
<h1 class='main-title'>
🏗️ Deteksi Retak Beton
</h1>

<p class='sub-title'>
Klasifikasi Crack dan No Crack Menggunakan Deep Learning
</p>
""", unsafe_allow_html=True)

# =====================
# LOAD MODEL
# =====================

MODEL_PATH = "model_crack_beton.h5"

model = load_model(MODEL_PATH)

# =====================
# CLASS
# =====================

class_names = [
    "Crack",
    "No Crack"
]

# =====================
# PREDICT
# =====================

def predict_image(img):

    img = img.resize((150,150))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    pred = model.predict(img_array, verbose=0)

    probs = tf.nn.softmax(pred[0]).numpy()

    pred_idx = np.argmax(probs)

    pred_class = class_names[pred_idx]

    confidence = probs[pred_idx] * 100

    return pred_class, confidence, probs

# =====================
# UPLOAD
# =====================

uploaded_file = st.file_uploader(
    "📤 Upload Foto Beton",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    pred_class, confidence, probs = predict_image(img)

    col1, col2 = st.columns([1.2,1])

    with col1:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.image(
            img,
            caption="Gambar Beton",
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        st.markdown("## 📊 Hasil Analisis")

        if pred_class == "Crack":

            st.markdown(
                f"<div class='pred-box crack'>Prediksi : {pred_class}</div>",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"<div class='pred-box normal'>Prediksi : {pred_class}</div>",
                unsafe_allow_html=True
            )

        st.write("")

        st.markdown(
            f"""
            <h2 style='color:#38bdf8'>
            Confidence : {confidence:.2f}%
            </h2>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

        st.write("")

        if pred_class == "Crack":

            st.error(
                "Terdeteksi indikasi retak pada permukaan beton."
            )

        else:

            st.success(
                "Permukaan beton terdeteksi dalam kondisi baik."
            )

        st.write("")

        st.markdown("### 📋 Probabilitas Semua Kelas")

        df = pd.DataFrame({
            "Kelas": class_names,
            "Probabilitas (%)":
            [round(x*100,2) for x in probs]
        })

        st.dataframe(
            df,
            use_container_width=True
        )

        st.bar_chart(
            df.set_index("Kelas")
        )

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown(f"""
    ### 🔎 Kesimpulan

    Berdasarkan hasil analisis citra beton menggunakan model Deep Learning,
    sistem mengklasifikasikan gambar ke kategori **{pred_class}**
    dengan tingkat keyakinan **{confidence:.2f}%**.
    """)

st.write("")
st.markdown("---")

st.markdown("""
<center>

Developed with ❤️ using TensorFlow & Streamlit

</center>
""", unsafe_allow_html=True)
