#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
from deepfake_detector import preprocess_image, compute_reconstruction_error, model, THRESHOLD

st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🧠 Deepfake Image Detector")
st.write("Upload an image to check whether it's real or AI-generated.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_array, image_display = preprocess_image(uploaded_file)

    st.image(image_display, caption="Uploaded Image", use_column_width=True)

    error = compute_reconstruction_error(model, image_array)
    st.write(f"🔍 Reconstruction Error: `{error:.5f}`")

    if error < THRESHOLD:
        st.success("✅ Prediction: This image is **Real**.")
    else:
        st.error("⚠️ Prediction: This image is **Fake**.")


# In[ ]:




