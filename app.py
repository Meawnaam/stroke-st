import streamlit as st
import os
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
    """
    Download model from Google Drive using gdown.
    Returns True if successful, False otherwise.
    """
    import gdown

    # Google Drive direct download URL สำหรับไฟล์ใหญ่
    url = f"https://drive.google.com/uc?id={FILE_ID}&export=download&confirm=t"

    st.info(f"📦 gdown version: {gdown.__version__}")

    try:
        with st.spinner("⏬ Downloading model (~500 MB) — please wait..."):
            output = gdown.download(
                id      = FILE_ID,   # ใช้ id= แทน url= เพื่อให้ gdown จัดการ confirmation เอง
                output  = MODEL_PATH,
                quiet   = False,
            )

        # ตรวจสอบผลลัพธ์
        if output is None:
            st.error("❌ gdown returned None — download failed.")
            return False

        if not os.path.exists(MODEL_PATH):
            st.error("❌ File not found after download.")
            return False

        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.info(f"📁 Downloaded file size: {size_mb:.1f} MB")

        if size_mb < 50:
            # ถ้าไฟล์เล็กกว่า 50MB แสดงว่าได้ HTML page มา ไม่ใช่ model
            with open(MODEL_PATH, "r", errors="ignore") as f:
                preview = f.read(500)
            st.error(
                f"❌ File too small ({size_mb:.2f} MB) — "
                "likely an HTML error page, not the model.\n\n"
                f"**File preview:**\n```\n{preview}\n```"
            )
            os.remove(MODEL_PATH)
            return False

        st.success(f"✅ Model downloaded! ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        st.error(f"❌ gdown error: {type(e).__name__}: {e}")
        st.exception(e)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False


# =====================================================
# LOAD MODEL (cached)
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf

    # Download ถ้าไม่มีไฟล์
    if not os.path.exists(MODEL_PATH):
        success = download_model()
        if not success:
            return None
    else:
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.info(f"📁 Using cached model ({size_mb:.1f} MB)")

    # Load model
    try:
        with st.spinner("🔄 Loading model into memory..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {type(e).__name__}: {e}")
        st.exception(e)
        # ลบไฟล์ที่อาจ corrupt แล้ว download ใหม่ครั้งหน้า
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
    return np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)


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
        "**ENSStrokeNet35** is a deep learning ensemble model "
        "for stroke detection from brain MRI images.\n\n"
        "⚠️ **Disclaimer:** For research/demo only. "
        "Not a substitute for professional medical advice."
    )
    st.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")
    st.markdown(f"**Input Size:** {IMG_SIZE}×{IMG_SIZE} px")
    st.markdown("---")

    # Debug info
    with st.expander("🔧 Debug Info"):
        import gdown
        st.code(
            f"gdown version : {gdown.__version__}\n"
            f"Model path    : {MODEL_PATH}\n"
            f"File exists   : {os.path.exists(MODEL_PATH)}\n"
            f"File size     : "
            f"{os.path.getsize(MODEL_PATH) / 1e6:.1f} MB"
            if os.path.exists(MODEL_PATH) else
            f"gdown version : {gdown.__version__}\n"
            f"Model path    : {MODEL_PATH}\n"
            f"File exists   : False"
        )

# Load model
model = load_model()

if model is None:
    st.error(
        "🚨 **Model could not be loaded.**\n\n"
        "Please verify:\n"
        "1. Google Drive File ID is correct\n"
        "2. File is shared → **'Anyone with the link'** can view\n"
        "3. Try clicking **'Rerun'** button above\n\n"
        f"📋 File ID: `{FILE_ID}`"
    )
    st.stop()

st.markdown("---")

# =====================================================
# UPLOADER & PREDICTION
# =====================================================
st.subheader("📤 Upload Brain MRI Image")

uploaded_file = st.file_uploader(
    "Choose a JPG or PNG file",
    type=["jpg", "jpeg", "png"],
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
            img_array = preprocess_image(uploaded_file)

            with st.spinner("🔍 Analyzing image..."):
                prediction = model.predict(img_array, verbose=0)

            # --- Interpret output ---
            if prediction.shape == 1:
                # Single sigmoid output
                prob            = float(prediction)
                predicted_index = 1 if prob >= 0.5 else 0
                confidence      = prob if predicted_index == 1 else 1 - prob
            else:
                # Softmax output
                predicted_index = int(np.argmax(prediction, axis=1))
                confidence      = float(prediction[predicted_index])

            predicted_class = (
                CLASS_NAMES[predicted_index]
                if predicted_index < len(CLASS_NAMES)
                else f"Class {predicted_index}"
            )

            # --- Display result ---
            if predicted_class == "Stroke":
                st.error(f"🔴 **{predicted_class} Detected**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.warning(
                    "⚠️ Stroke indicators found. "
                    "Please consult a doctor immediately."
                )
            else:
                st.success(f"🟢 **{predicted_class}**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.info("✅ This is a model prediction, not medical advice.")

            # --- Probability bars ---
            st.markdown("**All Class Probabilities:**")
            for i, name in enumerate(CLASS_NAMES):
                if i < prediction.shape:
                    p = float(prediction[i])
                    st.progress(p, text=f"{name}: {p * 100:.1f}%")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)

else:
    st.info("👆 Upload a brain MRI image to get a prediction.")
