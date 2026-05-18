import streamlit as st
import os
import numpy as np
import requests
from PIL import Image

# =====================================================
# PAGE CONFIG
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
CLASS_NAMES = ['Normal', 'Stroke']

# =====================================================
# HELPER: Fallback download with requests
# =====================================================
def _download_with_requests(model_path: str):
    """Fallback download for large Google Drive files."""

    session = requests.Session()
    URL = "https://docs.google.com/uc?export=download"

    # Step 1: Get confirmation token
    response = session.get(
        URL,
        params={"id": GOOGLE_DRIVE_FILE_ID, "confirm": 1},
        stream=True
    )

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    # Step 2: Download with token
    params = {"id": GOOGLE_DRIVE_FILE_ID, "confirm": token or "t"}
    response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    downloaded = 0
    progress_bar = st.progress(0, text="📥 Downloading model...")

    with open(model_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                mb = downloaded / (1024 * 1024)
                pct = min(downloaded / (500 * 1024 * 1024), 0.99)
                if downloaded % (10 * 1024 * 1024) < CHUNK_SIZE:
                    progress_bar.progress(pct, text=f"📥 Downloading... {mb:.0f} / ~500 MB")

    progress_bar.progress(1.0, text="✅ Download complete!")

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf

    model_path = os.path.join("/tmp", MODEL_FILENAME)

    # Download ถ้ายังไม่มีไฟล์
    if not os.path.exists(model_path):
        
        downloaded_ok = False
        
        # --- วิธีที่ 1: gdown ---
        try:
            import gdown
            st.info("⏬ Downloading model via gdown...")
            
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            
            # ตรวจสอบ version ของ gdown
            try:
                gdown_ver = tuple(int(x) for x in gdown.__version__.split(".")[:2])
                use_fuzzy = gdown_ver >= (4, 7)
            except Exception:
                use_fuzzy = False

            if use_fuzzy:
                gdown.download(url, model_path, quiet=False, fuzzy=True)
            else:
                gdown.download(url, model_path, quiet=False)

            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                if size_mb > 10:
                    downloaded_ok = True
                    st.success(f"✅ Downloaded via gdown ({size_mb:.1f} MB)")
                else:
                    st.warning(f"⚠️ gdown file too small ({size_mb:.1f} MB), trying fallback...")
                    os.remove(model_path)

        except Exception as e:
            st.warning(f"⚠️ gdown failed: {e} → Trying fallback...")

        # --- วิธีที่ 2: requests fallback ---
        if not downloaded_ok:
            try:
                st.info("⏬ Downloading model via requests (fallback)...")
                _download_with_requests(model_path)

                if os.path.exists(model_path):
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    if size_mb > 10:
                        downloaded_ok = True
                        st.success(f"✅ Downloaded via requests ({size_mb:.1f} MB)")
                    else:
                        st.error(f"❌ File too small ({size_mb:.1f} MB). Check Drive sharing.")
                        os.remove(model_path)
                        return None

            except Exception as e:
                st.error(f"❌ All download methods failed: {e}")
                return None

        if not downloaded_ok:
            st.error("❌ Could not download model.")
            return None

    # Load model
    try:
        with st.spinner("🔄 Loading model into memory..."):
            model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.exception(e)
        return None

# =====================================================
# PREPROCESSING
# =====================================================
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# =====================================================
# MAIN UI
# =====================================================
st.title("🧠 Stroke Detection")
st.subheader("Powered by ENSStrokeNet35")
st.markdown("---")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "**ENSStrokeNet35** is a deep learning model "
    "trained to detect stroke from brain MRI images.\n\n"
    "⚠️ **Disclaimer:** For research/demo purposes only. "
    "Not a substitute for professional medical diagnosis."
)
st.sidebar.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")
st.sidebar.markdown(f"**Input Size:** {IMG_SIZE}×{IMG_SIZE} px")

# Load model
model = load_model()

if model is None:
    st.error(
        "🚨 Model could not be loaded. Please check:\n"
        "1. Google Drive file ID is correct\n"
        "2. File is shared as **'Anyone with the link'**\n"
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
    help="Upload a brain MRI scan image"
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Uploaded Image:**")
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    with col2:
        st.markdown("**Prediction Result:**")
        try:
            import tensorflow as tf

            img_array = preprocess_image(uploaded_file)

            with st.spinner("🔍 Analyzing..."):
                prediction = model.predict(img_array, verbose=0)

            num_classes = prediction.shape

            if num_classes == 1:
                prob = float(prediction)
                predicted_index = 1 if prob >= 0.5 else 0
                confidence = prob if predicted_index == 1 else 1 - prob
            else:
                predicted_index = int(np.argmax(prediction, axis=1))
                confidence = float(prediction[predicted_index])

            predicted_class = (
                CLASS_NAMES[predicted_index]
                if predicted_index < len(CLASS_NAMES)
                else f"Class {predicted_index}"
            )

            # แสดงผล
            if predicted_class == 'Stroke':
                st.error(f"🔴 **{predicted_class}**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.warning("⚠️ Please consult a medical professional immediately.")
            else:
                st.success(f"🟢 **{predicted_class}**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.info("✅ This is a model prediction, not medical advice.")

            # Class probabilities
            st.markdown("**Class Probabilities:**")
            for i, name in enumerate(CLASS_NAMES):
                if i < prediction.shape:
                    prob_val = float(prediction[i])
                    st.progress(prob_val, text=f"{name}: {prob_val * 100:.1f}%")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)

else:
    st.info("👆 Please upload a brain MRI image to get started.")
