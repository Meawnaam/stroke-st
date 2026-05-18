import streamlit as st
import os
import numpy as np
from PIL import Image

# =====================================================
# PAGE CONFIG (ต้องเป็น st command แรกเสมอ)
# =====================================================
st.set_page_config(
    page_title="Stroke Detection - ENSStrokeNet35",
    page_icon="🧠",
    layout="centered"
)

# =====================================================
# CONSTANTS
# =====================================================
IMG_SIZE = (224, 224)
MODEL_FILENAME = "Imp_ENSStrokeNet35.keras"
GOOGLE_DRIVE_FILE_ID = "1PyI0XiQh7dZPj9_jq1h85uZMmbT43ZSS"

# Class names — ปรับให้ตรงกับ model ของคุณ
# ตรวจสอบจาก training: class_indices จาก flow_from_directory
# ปกติ flow_from_directory จะเรียงตาม alphabet
CLASS_NAMES = ['Normal', 'Stroke']  # <-- ปรับถ้าจำเป็น

# =====================================================
# LOAD MODEL (cached)
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Download model from Google Drive and load it."""
    import tensorflow as tf

    model_path = os.path.join("/tmp", MODEL_FILENAME)

    # ถ้ายังไม่มีไฟล์ใน /tmp ให้ download
    if not os.path.exists(model_path):
        try:
            import gdown
            with st.spinner("⏬ Downloading model (500MB)... Please wait..."):
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, model_path, quiet=False, fuzzy=True)

            # ตรวจสอบว่า download สำเร็จและไฟล์ถูกต้อง
            if not os.path.exists(model_path):
                st.error("❌ Download failed: File not found after download.")
                return None

            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if file_size_mb < 10:  # ถ้าไฟล์เล็กกว่า 10MB แสดงว่า download ผิดพลาด
                st.error(
                    f"❌ Downloaded file is too small ({file_size_mb:.1f} MB). "
                    "Likely a Google Drive error page, not the model file. "
                    "Please check that the file is shared publicly."
                )
                os.remove(model_path)  # ลบไฟล์ผิดออก
                return None

            st.success(f"✅ Model downloaded! ({file_size_mb:.1f} MB)")

        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            return None

    # Load model
    try:
        with st.spinner("🔄 Loading model into memory..."):
            model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.warning(
            "Possible causes:\n"
            "1. Model was saved with a different TensorFlow version\n"
            "2. Model file is corrupted\n"
            "3. Mixed precision layers incompatibility"
        )
        return None


# =====================================================
# PREPROCESSING
# =====================================================
def preprocess_image(uploaded_file):
    """Preprocess uploaded image for model prediction."""
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0          # Rescale [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim → (1, 224, 224, 3)
    return img_array


# =====================================================
# MAIN UI
# =====================================================
st.title("🧠 Stroke Detection")
st.subheader("Powered by ENSStrokeNet35")
st.markdown("---")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "**ENSStrokeNet35** is a deep learning model trained to detect "
    "stroke from brain MRI images.\n\n"
    "**How to use:**\n"
    "1. Wait for model to load\n"
    "2. Upload a brain MRI image\n"
    "3. View prediction result\n\n"
    "⚠️ **Disclaimer:** For research/demo purposes only. "
    "Not a substitute for professional medical diagnosis."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")
st.sidebar.markdown(f"**Input Size:** {IMG_SIZE}×{IMG_SIZE} px")

# Load model
model = load_model()

if model is None:
    st.error(
        "🚨 Model could not be loaded. Please check:\n"
        "1. Google Drive file ID is correct\n"
        "2. File is shared as 'Anyone with the link'\n"
        "3. Restart the app and try again"
    )
    st.stop()

st.markdown("---")

# =====================================================
# FILE UPLOADER & PREDICTION
# =====================================================
st.subheader("📤 Upload Brain MRI Image")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI scan image (JPG or PNG format)"
)

if uploaded_file is not None:
    # Layout: 2 columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Uploaded Image:**")
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

    with col2:
        st.markdown("**Prediction Result:**")

        try:
            import tensorflow as tf

            # Preprocess
            img_array = preprocess_image(uploaded_file)

            # Predict
            with st.spinner("🔍 Analyzing..."):
                prediction = model.predict(img_array, verbose=0)

            # ---- Interpret results ----
            num_classes = prediction.shape

            if num_classes == 2:
                # Binary classification (2 classes)
                predicted_index = int(np.argmax(prediction, axis=1))
                confidence = float(prediction[predicted_index])
                predicted_class = CLASS_NAMES[predicted_index]

            elif num_classes == 1:
                # Single sigmoid output
                prob = float(prediction)
                predicted_index = 1 if prob >= 0.5 else 0
                confidence = prob if predicted_index == 1 else 1 - prob
                predicted_class = CLASS_NAMES[predicted_index]

            else:
                # Multi-class (มากกว่า 2 class)
                predicted_index = int(np.argmax(prediction, axis=1))
                confidence = float(prediction[predicted_index])
                predicted_class = (
                    CLASS_NAMES[predicted_index]
                    if predicted_index < len(CLASS_NAMES)
                    else f"Class {predicted_index}"
                )

            # ---- Display result ----
            if predicted_class == 'Stroke':
                st.error(f"🔴 **{predicted_class}**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.warning(
                    "⚠️ Stroke indicators detected. "
                    "Please consult a medical professional immediately."
                )
            else:
                st.success(f"🟢 **{predicted_class}**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.info(
                    "✅ No stroke indicators detected. "
                    "This is a model prediction, not medical advice."
                )

            # Show all class probabilities
            st.markdown("**Class Probabilities:**")
            for i, name in enumerate(CLASS_NAMES):
                if i < prediction.shape:
                    prob_val = float(prediction[i])
                    st.progress(prob_val, text=f"{name}: {prob_val * 100:.1f}%")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.exception(e)  # Show full traceback for debugging

else:
    # Placeholder when no image uploaded
    st.info("👆 Please upload a brain MRI image to get started.")
