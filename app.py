import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title = "Stroke Detection - ENSStrokeNet35",
    page_icon  = "🧠",
    layout     = "centered"
)

IMG_SIZE         = (224, 224)
MODEL_FILENAME   = "Imp_ENSStrokeNet35.keras"
FILE_ID          = "1PyI0XiQh7dZPj9_jq1h85uZMmbT43ZSS"
MODEL_PATH       = f"/tmp/{MODEL_FILENAME}"
CLASS_NAMES      = ['no_stroke', 'stroke']
STROKE_THRESHOLD = 0.55


def download_model() -> bool:
    import gdown
    try:
        with st.spinner("⏬ Downloading model (~500 MB)..."):
            output = gdown.download(
                id     = FILE_ID,
                output = MODEL_PATH,
                quiet  = False,
            )
        if output is None or not os.path.exists(MODEL_PATH):
            st.error("❌ Download failed.")
            return False
        
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        if size_mb < 50:
            with open(MODEL_PATH, "r", errors="ignore") as f:
                preview = f.read(300)
            
            # แยก st.error กับ st.code ออกจากกันเพื่อป้องกัน SyntaxError จากเครื่องหมายฟันหนู
            st.error(f"❌ File too small ({size_mb:.2f} MB). The downloaded file might be a Google Drive error page.")
            st.code(preview, language="text")
            
            os.remove(MODEL_PATH)
            return False
            
        st.success(f"✅ Downloaded! ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        st.error(f"❌ Download error: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False


@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    import keras

    try:
        mixed_precision.set_global_policy('mixed_float16')
    except Exception:
        pass

    if not os.path.exists(MODEL_PATH):
        if not download_model():
            return None
    
    try:
        with st.spinner("🔄 Loading model..."):
            m = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded!")
        return m
    except Exception as e1:
        st.warning(f"⚠️ Try 1 failed: {e1}")

    try:
        with st.spinner("🔄 Loading model (attempt 2)..."):
            m = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model loaded!")
        return m
    except Exception as e2:
        st.warning(f"⚠️ Try 2 failed: {e2}")

    try:
        with st.spinner("🔄 Loading model (attempt 3)..."):
            m = keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model loaded!")
        return m
    except Exception as e3:
        st.warning(f"⚠️ Try 3 failed: {e3}")

    st.error("❌ All loading attempts failed.")
    return None


def preprocess_image(uploaded_file) -> np.ndarray:
    img       = Image.open(uploaded_file).convert("RGB")
    img       = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr       = np.array(img, dtype=np.float32) / 255.0        
    arr       = np.expand_dims(arr, axis=0)
    return arr


def predict(model, img_array: np.ndarray) -> dict:
    raw  = model.predict(img_array, verbose=0)
    pred = raw.astype(np.float32).flatten()

    # สำหรับโมเดลแบบ Binary (1 Node)
    p1 = float(pred[0])
    p0 = float(1.0 - p1) 

    p0 = float(np.clip(p0, 0.0, 1.0))
    p1 = float(np.clip(p1, 0.0, 1.0))

    idx   = 1 if p1 > STROKE_THRESHOLD else 0
    label = CLASS_NAMES[idx]
    conf  = p1 if idx == 1 else p0

    return {
        "predicted_class" : label,
        "predicted_index" : idx,
        "confidence"      : conf,
        "stroke_prob"     : p1,
        "no_stroke_prob"  : p0,
        "raw_shape"       : str(raw.shape),
        "raw_values"      : raw.astype(np.float32).tolist(),
    }


# ── UI ──────────────────────────────────────────────
st.title("🧠 Stroke Detection")
st.subheader("Powered by ENSStrokeNet35")
st.markdown("---")

with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "**ENSStrokeNet35** — deep learning ensemble model "
        "for stroke detection from brain MRI images.\n\n"
        "⚠️ For research/demo only. "
        "Not a substitute for medical advice."
    )
    st.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")
    st.markdown(f"**Input Size:** {IMG_SIZE} × {IMG_SIZE} px")
    st.markdown(f"**Stroke Threshold:** `{STROKE_THRESHOLD}`")
    st.markdown("---")

# โหลดโมเดลก่อนทำ Debug และการทำงานอื่นๆ
model = load_model()

if model is None:
    st.error(
        "🚨 Model could not be loaded.\n\n"
        "1. Check Google Drive File ID\n"
        "2. File must be shared as **'Anyone with the link'**\n"
        "3. Click **Rerun**\n\n"
        "4. หรือเกิดจาก RAM บนระบบ Cloud เต็ม (เนื่องจากโมเดลมีขนาดใหญ่มาก)\n\n"
        f"File ID: `{FILE_ID}`"
    )
    st.stop()

# แสดง Debug ข้อมูลเมื่อโหลดโมเดลเรียบร้อยแล้ว
with st.sidebar:
    with st.expander("🔧 Debug Info"):
        exists = os.path.exists(MODEL_PATH)
        if exists:
            st.code(
                f"FILE        : {MODEL_PATH}\n"
                f"exists      : {exists}\n"
                f"size        : {os.path.getsize(MODEL_PATH)/1e6:.1f} MB"
            )
        else:
            st.code(f"FILE        : {MODEL_PATH}\nexists      : False")

    with st.expander("📊 Model Info"):
        try:
            st.code(
                f"Input  : {model.input_shape}\n"
                f"Output : {model.output_shape}\n"
                f"Params : {model.count_params():,}"
            )
        except Exception:
            st.write("Unavailable.")

st.markdown("---")
st.subheader("📤 Upload Brain MRI Image")

uploaded_file = st.file_uploader(
    "Choose JPG or PNG",
    type=["jpg", "jpeg", "png"],
)

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
            arr = preprocess_image(uploaded_file)

            with st.spinner("🔍 Analyzing..."):
                result = predict(model, arr)

            label          = result["predicted_class"]
            conf           = result["confidence"]
            stroke_prob    = result["stroke_prob"]
            no_stroke_prob = result["no_stroke_prob"]

            if label == "stroke":
                st.error("🔴 **Stroke Detected**")
                st.metric("Confidence", f"{conf * 100:.2f}%")
                st.warning(
                    "⚠️ Stroke indicators detected.\n\n"
                    "Please consult a doctor immediately."
                )
            else:
                st.success("🟢 **No Stroke Detected**")
                st.metric("Confidence", f"{conf * 100:.2f}%")
                st.info("✅ This is a model prediction, not medical advice.")

            st.markdown("**Class Probabilities:**")
            st.progress(
                float(no_stroke_prob),
                text=f"No Stroke : {no_stroke_prob * 100:.2f}%"
            )
            st.progress(
                float(stroke_prob),
                text=f"Stroke    : {stroke_prob * 100:.2f}%"
            )
            st.caption(f"Threshold: `{STROKE_THRESHOLD}`")

            with st.expander("🔍 Raw Details"):
                st.json({
                    "shape"        : result["raw_shape"],
                    "values"       : result["raw_values"],
                    "no_stroke"    : f"{no_stroke_prob:.6f}",
                    "stroke"       : f"{stroke_prob:.6f}",
                    "threshold"    : STROKE_THRESHOLD,
                    "is_stroke"    : stroke_prob > STROKE_THRESHOLD,
                })

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)

else:
    st.info("👆 Upload a brain MRI image to get started.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1️⃣ Upload**\n\nChoose a brain MRI (JPG/PNG)")
    with c2:
        st.markdown("**2️⃣ Analyze**\n\nModel processes automatically")
    with c3:
        st.markdown("**3️⃣ Result**\n\nGet prediction + confidence")
