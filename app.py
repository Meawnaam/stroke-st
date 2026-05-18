import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # บอก Keras 3 ให้ใช้ TF backend

import streamlit as st
import numpy as np
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
IMG_SIZE       = (224, 224)
MODEL_FILENAME = "Imp_ENSStrokeNet35.keras"
FILE_ID        = "1PyI0XiQh7dZPj9_jq1h85uZMmbT43ZSS"
MODEL_PATH     = f"/tmp/{MODEL_FILENAME}"
CLASS_NAMES    = ['Normal', 'Stroke']

# =====================================================
# DOWNLOAD MODEL
# =====================================================
def download_model() -> bool:
    import gdown

    st.info(f"📦 gdown version: {gdown.__version__}")

    try:
        with st.spinner("⏬ Downloading model (~500 MB) — please wait..."):
            output = gdown.download(
                id     = FILE_ID,
                output = MODEL_PATH,
                quiet  = False,
            )

        if output is None or not os.path.exists(MODEL_PATH):
            st.error("❌ Download failed — output is None.")
            return False

        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.info(f"📁 Downloaded file size: {size_mb:.1f} MB")

        if size_mb < 50:
            with open(MODEL_PATH, "r", errors="ignore") as f:
                preview = f.read(300)
            st.error(
                f"❌ File too small ({size_mb:.2f} MB) — "
                f"likely an HTML error page.\n\n"
                f"**Preview:**\n```\n{preview}\n```"
            )
            os.remove(MODEL_PATH)
            return False

        st.success(f"✅ Downloaded successfully! ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        st.error(f"❌ Download error: {type(e).__name__}: {e}")
        st.exception(e)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False


# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf
    import keras

    st.info(
        f"🔧 TensorFlow: `{tf.__version__}` | "
        f"Keras: `{keras.__version__}`"
    )

    # Download ถ้าไม่มีไฟล์
    if not os.path.exists(MODEL_PATH):
        success = download_model()
        if not success:
            return None
    else:
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.info(f"📁 Using cached model ({size_mb:.1f} MB)")

    # --- โหลด Model ---
    # วิธีที่ 1: keras.models.load_model (Keras 3 native)
    try:
        with st.spinner("🔄 Loading model (attempt 1/3 — keras native)..."):
            model = keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded! (keras native)")
        return model
    except Exception as e1:
        st.warning(f"⚠️ Attempt 1 failed: {e1}")

    # วิธีที่ 2: keras โหลดแบบ compile=False
    try:
        with st.spinner("🔄 Loading model (attempt 2/3 — no compile)..."):
            model = keras.models.load_model(
                MODEL_PATH,
                compile=False
            )
        st.success("✅ Model loaded! (no compile)")
        return model
    except Exception as e2:
        st.warning(f"⚠️ Attempt 2 failed: {e2}")

    # วิธีที่ 3: tf.keras.models.load_model
    try:
        with st.spinner("🔄 Loading model (attempt 3/3 — tf.keras)..."):
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )
        st.success("✅ Model loaded! (tf.keras)")
        return model
    except Exception as e3:
        st.warning(f"⚠️ Attempt 3 failed: {e3}")

    st.error("❌ All loading attempts failed.")
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    return None


# =====================================================
# PREPROCESSING
# =====================================================
def preprocess_image(uploaded_file) -> np.ndarray:
    img       = Image.open(uploaded_file).convert("RGB")
    img       = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)  # → (1, 224, 224, 3)


# =====================================================
# MAIN UI
# =====================================================
st.title("🧠 Stroke Detection")
st.subheader("Powered by ENSStrokeNet35")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "**ENSStrokeNet35** detects stroke indicators "
        "from brain MRI images using deep learning.\n\n"
        "⚠️ **Disclaimer:** For research/demo only. "
        "Not a substitute for professional medical advice."
    )
    st.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")
    st.markdown(f"**Input Size:** {IMG_SIZE}×{IMG_SIZE} px")
    st.markdown("---")

    with st.expander("🔧 Debug Info"):
        exists    = os.path.exists(MODEL_PATH)
        size_info = (
            f"{os.path.getsize(MODEL_PATH)/1e6:.1f} MB"
            if exists else "N/A"
        )
        st.code(
            f"MODEL_PATH  : {MODEL_PATH}\n"
            f"File exists : {exists}\n"
            f"File size   : {size_info}"
        )

# =====================================================
# LOAD MODEL
# =====================================================
model = load_model()

if model is None:
    st.error(
        "🚨 **Model could not be loaded.**\n\n"
        "Please verify:\n"
        "1. Google Drive File ID is correct\n"
        "2. File shared as **'Anyone with the link'**\n"
        "3. Click **Rerun** to try again\n\n"
        f"📋 File ID used: `{FILE_ID}`"
    )
    st.stop()

# แสดง model summary สั้นๆ
with st.sidebar:
    with st.expander("📊 Model Info"):
        st.code(
            f"Input shape  : {model.input_shape}\n"
            f"Output shape : {model.output_shape}\n"
            f"Total params : {model.count_params():,}"
        )

st.markdown("---")

# =====================================================
# UPLOADER & PREDICTION
# =====================================================
st.subheader("📤 Upload Brain MRI Image")

uploaded_file = st.file_uploader(
    "Choose a JPG or PNG file",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI scan image for stroke detection"
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Uploaded Image:**")
        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    with col2:
        st.markdown("**Prediction Result:**")
        try:
            img_array  = preprocess_image(uploaded_file)

            with st.spinner("🔍 Analyzing image..."):
                prediction = model.predict(img_array, verbose=0)

            # --- Interpret output ---
            if prediction.shape == 1:
                # Single sigmoid
                prob            = float(prediction)
                predicted_index = 1 if prob >= 0.5 else 0
                confidence      = prob if predicted_index == 1 else 1 - prob
            else:
                # Softmax
                predicted_index = int(np.argmax(prediction, axis=1))
                confidence      = float(prediction[predicted_index])

            predicted_class = (
                CLASS_NAMES[predicted_index]
                if predicted_index < len(CLASS_NAMES)
                else f"Class {predicted_index}"
            )

            # --- Show result ---
            if predicted_class == "Stroke":
                st.error(f"🔴 **{predicted_class} Detected**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.warning(
                    "⚠️ Stroke indicators detected.\n"
                    "Please consult a medical professional immediately."
                )
            else:
                st.success(f"🟢 **{predicted_class}**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.info(
                    "✅ No stroke indicators detected.\n"
                    "This is a model prediction, not medical advice."
                )

            # --- Probability bars ---
            st.markdown("**Class Probabilities:**")
            for i, name in enumerate(CLASS_NAMES):
                if i < prediction.shape:
                    p = float(prediction[i])
                    st.progress(
                        value = p,
                        text  = f"{name}: {p * 100:.1f}%"
                    )

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)

else:
    st.info("👆 Upload a brain MRI image to get started.")
