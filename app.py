{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "afe3ef89-6f22-4485-a9b2-0611c0fcf845",
   "metadata": {},
   "outputs": [],
   "source": [
    "###python APP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "52a635a9-3c57-42ac-bd4e-580d92dfd312",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing.image import img_to_array\n",
    "import os\n",
    "\n",
    "# --- SETTINGS ---\n",
    "IMAGE_SIZE = 224\n",
    "MODEL_PATH = \"autoencoder_model.h5\"  # Ensure this file is in the same directory or update the path\n",
    "THRESHOLD = 0.02  # Update this based on your notebook evaluation\n",
    "\n",
    "# --- LOAD MODEL ---\n",
    "@st.cache_resource\n",
    "def load_autoencoder_model():\n",
    "    if not os.path.exists(MODEL_PATH):\n",
    "        st.error(f\"Model file not found at: {MODEL_PATH}\")\n",
    "        st.stop()\n",
    "    model = load_model(MODEL_PATH)\n",
    "    return model\n",
    "\n",
    "model = load_autoencoder_model()\n",
    "\n",
    "# --- IMAGE PROCESSING ---\n",
    "def preprocess_image(uploaded_file):\n",
    "    try:\n",
    "        image = Image.open(uploaded_file).convert(\"RGB\")\n",
    "        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))\n",
    "        image_array = img_to_array(image) / 255.0\n",
    "        image_array = np.expand_dims(image_array, axis=0)\n",
    "        return image_array, image\n",
    "    except Exception as e:\n",
    "        st.error(f\"Error processing image: {e}\")\n",
    "        st.stop()\n",
    "\n",
    "def compute_reconstruction_error(model, image_array):\n",
    "    reconstructed = model.predict(image_array)\n",
    "    mse = np.mean(np.square(image_array - reconstructed))\n",
    "    return mse\n",
    "\n",
    "# --- STREAMLIT UI ---\n",
    "st.set_page_config(page_title=\"Deepfake Detector\", layout=\"centered\")\n",
    "st.title(\"🧠 Deepfake Image Detector\")\n",
    "st.markdown(\"Upload an image to check whether it's **Real** or **AI-generated (Fake)** using an Autoencoder model.\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"📤 Upload Image\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "if uploaded_file:\n",
    "    image_array, image_display = preprocess_image(uploaded_file)\n",
    "    st.image(image_display, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "    error = compute_reconstruction_error(model, image_array)\n",
    "    st.info(f\"🔍 Reconstruction Error: `{error:.5f}`\")\n",
    "\n",
    "    if error < THRESHOLD:\n",
    "        st.success(\"✅ Prediction: This image is **Real**.\")\n",
    "    else:\n",
    "        st.error(\"⚠️ Prediction: This image is **Fake**.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
