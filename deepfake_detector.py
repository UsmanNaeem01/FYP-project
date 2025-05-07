#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# Load the model once
model = load_model("autoencoder_model.h5", compile=False)

# Set the threshold (example: adjust according to your real training results)
THRESHOLD = 0.0015  # Replace this with your actual threshold

def preprocess_image(uploaded_file, image_size=224):
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((image_size, image_size))
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image  # Array for model, PIL image for display

def compute_reconstruction_error(model, image_array):
    reconstructed = model.predict(image_array)
    error = np.mean(np.square(image_array - reconstructed))
    return error


# In[ ]:




