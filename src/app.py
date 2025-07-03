
import streamlit as st
import numpy as np
import tensorflow as tf
from data_preprocessing import preprocess_image

st.set_page_config(page_title="Verificação de Assinaturas", layout="centered")
st.title("Verificação de Assinaturas Manuscritas")
st.write("Compare duas imagens de assinaturas e descubra se pertencem à mesma pessoa.")

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

model = load_model("siamese_signature_model.h5")

uploaded_file1 = st.file_uploader("Selecione a primeira assinatura", type=["png", "jpg", "jpeg", "bmp"], key="file1")
uploaded_file2 = st.file_uploader("Selecione a segunda assinatura", type=["png", "jpg", "jpeg", "bmp"], key="file2")

if uploaded_file1 and uploaded_file2:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp1, tempfile.NamedTemporaryFile(delete=False) as tmp2:
        tmp1.write(uploaded_file1.read())
        tmp2.write(uploaded_file2.read())
        img1 = preprocess_image(tmp1.name)
        img2 = preprocess_image(tmp2.name)
        if img1.ndim == 2:
            img1 = np.expand_dims(img1, axis=-1)
        if img2.ndim == 2:
            img2 = np.expand_dims(img2, axis=-1)
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        dist = model.predict([img1, img2])[0][0]
        st.image([tmp1.name, tmp2.name], caption=["Assinatura 1", "Assinatura 2"], width=200)
        st.write(f"Distância calculada: {dist:.4f}")
        threshold = 0.5
        if dist < threshold:
            st.success("Assinaturas COMPATÍVEIS (provavelmente da mesma pessoa)")
        else:
            st.error("Assinaturas DIFERENTES (provavelmente de pessoas diferentes)")
else:
    st.info("Faça upload de duas imagens para comparar.")
