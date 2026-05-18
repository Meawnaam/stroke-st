import streamlit as st
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests
import zipfile
import io

# --- Configuration ---
# Set mixed precision policy early if using GPU,
# otherwise, it might not be beneficial on CPU and could cause issues.
# For Streamlit Cloud (CPU-only usually), mixed_float16 might not offer much benefit
# and can sometimes cause compatibility issues if not carefully managed.
# It's often safer to stick to float32 on CPU-only deployments unless you
# specifically test and confirm its stability and performance gains.
# For this example, I'll keep it commented out for simplicity unless
# you know your model absolutely requires it and the environment supports it.
# mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = (224, 224)
MODEL_FILENAME = 'Imp_ENSStrokeNet35.keras'

# --- Google Drive Link for Model Download ---
GOOGLE_DRIVE_FILE_ID = '1PyI0XiQh7dZPj9_jq1h85uZMmbT43ZSS' # <--- IMPORTANT: Update this!

# Cache the model loading to ensure it's loaded only once per session
@st.cache_resource
def load_keras_model(file_id):
    """Downloads and loads the Keras model from Google Drive."""
    st.write("Downloading model... This may take a moment.")
    try:
        # Construct the direct download URL for Google Drive
        # This URL bypasses browser UI and initiates direct download
        drive_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        response = requests.get(drive_url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors

        # Save the model to a temporary file
        model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        st.success(f"Model downloaded to {model_path}")

        # Load the model
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model from Google Drive: {e}")
        st.warning("Please ensure the Google Drive file ID is correct and the file is shared publicly ('Anyone with the link').")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Make sure your Keras model is compatible with the TensorFlow version used by Streamlit Cloud.")
        return None

# --- Main Streamlit App ---
st.title("Stroke Prediction with ENSStrokeNet35")
st.write("Upload a brain MRI image to get a stroke prediction.")

# Load the model
model = load_keras_model(GOOGLE_DRIVE_FILE_ID)

if model is None:
    st.stop() # Stop if model couldn't be loaded

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    try:
        img = image.load_img(uploaded_file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Create a batch
        img_array = img_array / 255.0 # Rescale as per your test_datagen

        # Make prediction
        prediction = model.predict(img_array)
        
        # Interpret prediction (assuming categorical output for simplicity)
        class_names = ['Normal', 'Stroke'] # Adjust based on your actual class names
        
        # If your model outputs probabilities for multiple classes:
        predicted_class_index = np.argmax(prediction, axis=1)
        confidence = prediction[predicted_class_index]
        predicted_class_name = class_names[predicted_class_index]

        st.subheader(f"Prediction: **{predicted_class_name}**")
        st.write(f"Confidence: {confidence:.2f}")

        if predicted_class_name == 'Stroke':
            st.error("Likely Stroke Detected! Please consult a medical professional immediately.")
        else:
            st.success("No Stroke Detected. However, this is a model prediction, not medical advice.")

    except Exception as e:
        st.error(f"Error processing image or making prediction: {e}")
        st.write("Please ensure the uploaded image is valid and try again.")

st.sidebar.header("About")
st.sidebar.info(
    "This Streamlit application uses a pre-trained Keras model (ENSStrokeNet35) "
    "to predict the presence of stroke from brain MRI images. "
    "The model is downloaded from Google Drive at runtime."
    "\n\n**Disclaimer:** This tool is for demonstration purposes only and "
    "should not be used as a substitute for professional medical advice."
)
