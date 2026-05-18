import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import numpy as np
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title = "Stroke Detection - ENSStrokeNet35",
    page_icon  = "🧠",
    layout     = "centered"
)

# =====================================================
# CONSTANTS
# =====================================================
IMG_SIZE         = (224, 224)
MODEL_FILENAME   = "Imp_ENSStrokeNet35.keras"
FILE_ID          = "1PyI0XiQh7dZPj9_jq1h85uZMmbT43ZSS"
MODEL_PATH       = f"/tmp/{MODEL_FILENAME}"
CLASS_NAMES      = ['no_stroke', 'stroke']
STROKE_THRESHOLD = 0.55

# =====================================================
# DOWNLOAD MODEL
# =====================================================
def download_model() -> bool:
    """Download model from Google Drive using gdown."""
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
            st.error("❌ Download failed — file not found.")
            return False

        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.info(f"📁 Downloaded file size: {size_mb:.1f} MB")

        if size_mb < 50:
            with open(MODEL_PATH, "r", errors="ignore") as f:
                preview = f.read(300)
            st.error(
                f"❌ File too small ({size_mb:.2f} MB)\n\n"
                f"**Preview:**\n```\n{preview}\n```\n\n"
                "Please share the file as **'Anyone with the link'**."
            )
            os.remove(MODEL_PATH)
            return False

        st.success(f"✅ Downloaded! ({size_mb:.1f} MB)")
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
    """Load model with mixed_float16 — same as training."""
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    import keras

    st.info(
        f"🔧 TensorFlow: `{tf.__version__}` | "
        f"Keras: `{keras.__version__}`"
    )

    mixed_precision.set_global_policy('mixed_float16')
    st.info("🔧 Mixed precision: `mixed_float16`")

    if not os.path.exists(MODEL_PATH):
        success = download_model()
        if not success:
            return None
    else:
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.info(f"📁 Using cached model ({size_mb:.1f} MB)")

    # Attempt 1: tf.keras
    try:
        with st.spinner("🔄 Loading model (1/3 — tf.keras)..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded! (tf.keras)")
        return model
    except Exception as e1:
        st.warning(f"⚠️ Attempt 1 failed: {e1}")

    # Attempt 2: tf.keras compile=False
    try:
        with st.spinner("🔄 Loading model (2/3 — compile=False)..."):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model loaded! (compile=False)")
        return model
    except Exception as e2:
        st.warning(f"⚠️ Attempt 2 failed: {e2}")

    # Attempt 3: keras native
    try:
        with st.spinner("🔄 Loading model (3/3 — keras native)..."):
            model = keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model loaded! (keras native)")
        return model
    except Exception as e3:
        st.warning(f"⚠️ Attempt 3 failed: {e3}")

    st.error("❌ All loading attempts failed.")
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    return None


# =====================================================
# PREPROCESSING — ตรงกับ training
# =====================================================
def preprocess_image(uploaded_file) -> np.ndarray:
    """
    Preprocess exactly as training:
      - Resize  : (224, 224)
      - Rescale : / 255.0
      - dtype   : float16
      - Shape   : (1, 224, 224, 3)
    """
    img       = Image.open(uploaded_file).convert("RGB")
    img       = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = img_array.astype('float16')
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# =====================================================
# PREDICT — confirmed shape (1, 2)
# =====================================================
def predict(model, img_array: np.ndarray) -> dict:
    """
    Predict — output shape: (1, 2)
    index 0 = no_stroke
    index 1 = stroke
    """
    raw_predictions = model.predict(img_array, verbose=0)

    # float16 → float32
    predictions = raw_predictions.astype(np.float32)

    # (1, 2) → 1D array [no_stroke_prob, stroke_prob]
    flat = predictions.flatten()

    # ดึงแต่ละค่าด้วย index ชัดเจน
    no_stroke_prob = float(flat)   # ← 
    stroke_prob    = float(flat)   # ← 

    # Clamp [0, 1]
    stroke_prob    = float(np.clip(stroke_prob,    0.0, 1.0))
    no_stroke_prob = float(np.clip(no_stroke_prob, 0.0, 1.0))

    # Threshold 0.55
    predicted_index = 1 if stroke_prob > STROKE_THRESHOLD else 0
    predicted_class = CLASS_NAMES[predicted_index]
    confidence      = stroke_prob if predicted_index == 1 else no_stroke_prob

    return {
        "predicted_class" : predicted_class,
        "predicted_index" : predicted_index,
        "confidence"      : confidence,
        "stroke_prob"     : stroke_prob,
        "no_stroke_prob"  : no_stroke_prob,
        "raw_shape"       : str(predictions.shape),
        "raw_values"      : predictions.tolist(),
    }
# =====================================================
# MAIN UI
# =====================================================
st.title("🧠 Stroke Detection")
st.subheader("Powered by ENSStrokeNet35")
st.markdown("---")

# ----- Sidebar -----
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "**ENSStrokeNet35** is a deep learning ensemble model "
        "for stroke detection from brain MRI images.\n\n"
        "⚠️ **Disclaimer:** For research and demo purposes only. "
        "Not a substitute for professional medical advice."
    )
    st.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")
    st.markdown(f"**Input Size:** {IMG_SIZE} × {IMG_SIZE} px")
    st.markdown(f"**Stroke Threshold:** `{STROKE_THRESHOLD}`")
    st.markdown("---")

    with st.expander("🔧 Debug Info"):
        exists    = os.path.exists(MODEL_PATH)
        size_info = (
            f"{os.path.getsize(MODEL_PATH) / 1e6:.1f} MB"
            if exists else "N/A"
        )
        st.code(
            f"MODEL_PATH  : {MODEL_PATH}\n"
            f"File exists : {exists}\n"
            f"File size   : {size_info}"
        )

# ----- Load Model -----
model = load_model()

if model is None:
    st.error(
        "🚨 **Model could not be loaded.**\n\n"
        "Please check:\n"
        "1. Google Drive File ID is correct\n"
        "2. File is shared as **'Anyone with the link'**\n"
        "3. Click **Rerun** to try again\n\n"
        f"📋 File ID: `{FILE_ID}`"
    )
    st.stop()

with st.sidebar:
    with st.expander("📊 Model Info"):
        try:
            st.code(
                f"Input  : {model.input_shape}\n"
                f"Output : {model.output_shape}\n"
                f"Params : {model.count_params():,}"
            )
        except Exception:
            st.write("Model info unavailable.")

st.markdown("---")

# =====================================================
# FILE UPLOADER
# =====================================================
st.subheader("📤 Upload Brain MRI Image")

uploaded_file = st.file_uploader(
    label = "Choose a JPG or PNG file",
    type  = ["jpg", "jpeg", "png"],
    help  = "Upload a brain MRI scan image for stroke detection",
)

# =====================================================
# PREDICTION & RESULTS
# =====================================================
if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Uploaded Image:**")
        st.image(
            uploaded_file,
            caption             = uploaded_file.name,
            use_container_width = True,
        )

    with col2:
        st.markdown("**Prediction Result:**")

        try:
            img_array = preprocess_image(uploaded_file)

            with st.spinner("🔍 Analyzing image..."):
                result = predict(model, img_array)

            predicted_class = result["predicted_class"]
            confidence      = result["confidence"]
            stroke_prob     = result["stroke_prob"]
            no_stroke_prob  = result["no_stroke_prob"]

            # Result
            if predicted_class == "stroke":
                st.error("🔴 **Stroke Detected**")
                st.metric(
                    label = "Confidence",
                    value = f"{confidence * 100:.2f}%"
                )
                st.warning(
                    "⚠️ Stroke indicators detected.\n\n"
                    "Please consult a medical professional immediately."
                )
            else:
                st.success("🟢 **No Stroke Detected**")
                st.metric(
                    label = "Confidence",
                    value = f"{confidence * 100:.2f}%"
                )
                st.info(
                    "✅ No stroke indicators detected.\n\n"
                    "This is a model prediction, not medical advice."
                )

            # Probability bars
            st.markdown("**Class Probabilities:**")
            st.progress(
                value = float(no_stroke_prob),
                text  = f"No Stroke : {no_stroke_prob * 100:.2f}%"
            )
            st.progress(
                value = float(stroke_prob),
                text  = f"Stroke    : {stroke_prob * 100:.2f}%"
            )
            st.caption(
                f"ℹ️ Stroke threshold: `{STROKE_THRESHOLD}` "
                f"(stroke if prob > {STROKE_THRESHOLD})"
            )

            # Raw debug
            with st.expander("🔍 Raw Prediction Details"):
                st.json({
                    "output_shape"   : result["raw_shape"],
                    "raw_values"     : result["raw_values"],
                    "no_stroke_prob" : f"{no_stroke_prob:.6f}",
                    "stroke_prob"    : f"{stroke_prob:.6f}",
                    "threshold"      : STROKE_THRESHOLD,
                    "decision"       : (
                        f"stroke_prob ({stroke_prob:.4f}) > "
                        f"threshold ({STROKE_THRESHOLD}) = "
                        f"{stroke_prob > STROKE_THRESHOLD}"
                    ),
                })

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)

else:
    st.info("👆 Please upload a brain MRI image to get started.")
    st.markdown("---")
    st.markdown("#### How it works:")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            "**1️⃣ Upload**\n\n"
            "Choose a brain MRI image (JPG/PNG)"
        )
    with col_b:
        st.markdown(
            "**2️⃣ Analyze**\n\n"
            "Model processes the image automatically"
        )
    with col_c:
        st.markdown(
            "**3️⃣ Result**\n\n"
            "Get prediction with confidence score"
        )
