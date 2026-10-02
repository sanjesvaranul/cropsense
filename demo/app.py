import io
from pathlib import Path
import requests
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="CropSense - Mobile Crop, Stage & Disease Intelligence",
    page_icon="🌾",
    layout="centered"
)

# Custom header styling
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stMetric {
        background-color: #f0f4f1;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 CropSense")
st.subheader("Mobile Crop, Stage & Disease Intelligence (Task 4)")
st.write(
    "Edge-deployable multi-task vision intelligence running **EfficientNet-B0** "
    "for simultaneous crop type, growth stage identification, and foliar disease diagnosis."
)

# Disease knowledge base / agronomic advisory
ADVISORIES = {
    "Healthy": {
        "status": "success",
        "icon": "🟢",
        "desc": "Crop appears healthy with no visible signs of foliar lesion or pathogen damage.",
        "action": "Maintain balanced N-P-K fertilization, regular scouting, and optimal irrigation management."
    },
    "Bacterial Leaf Blight": {
        "status": "error",
        "icon": "🔴",
        "desc": "Bacterial infection caused by Xanthomonas oryzae pv. oryzae.",
        "action": "Avoid excess nitrogen fertilizer. Maintain field drainage. Consider copper hydroxide or validated bactericide sprays during early stages."
    },
    "Brown Spot": {
        "status": "warning",
        "icon": "🟠",
        "desc": "Fungal pathogen caused by Bipolaris oryzae; frequently correlated with nutrient-poor or stressed soil.",
        "action": "Correct potassium and micronutrient deficiencies (Zinc/Silicon). Apply protective fungicides (Mancozeb, Carbendazim, or Tricyclazole) as indicated."
    },
    "Leaf Blast": {
        "status": "error",
        "icon": "🔴",
        "desc": "Aggressive fungal infection caused by Magnaporthe oryzae (spindle-shaped lesions).",
        "action": "Urgent intervention: Apply systemic fungicide (Tricyclazole 75 WP or Isoprothiolane). Avoid high nitrogen top-dressing and stagnant cool water."
    },
    "Leaf Smut": {
        "status": "warning",
        "icon": "🟠",
        "desc": "Foliar fungal infection caused by Entyloma oryzae (small, angular black spots).",
        "action": "Usually prevalent during late-season maturation. If severe, apply foliar propiconazole or copper oxychloride; avoid dense canopy humidity."
    }
}

# Sidebar for Sample Images / Presets
st.sidebar.header("🧪 Test Sample Library")
sample_dir = Path("data/external/rice_diseases")
sample_options = {"None (Upload own)": None}

if sample_dir.exists():
    for sub in sample_dir.iterdir():
        if sub.is_dir():
            imgs = list(sub.glob("*.jpg")) + list(sub.glob("*.png"))
            if imgs:
                sample_options[f"Rice ({sub.name})"] = str(imgs[0])

# Add a healthy sample if available
raw_rice = list(Path("data/raw/images").glob("*.jpg"))
if raw_rice:
    sample_options["Rice (Healthy field sample)"] = str(raw_rice[0])

selected_sample = st.sidebar.selectbox("Choose a sample to test:", list(sample_options.keys()))

uploaded = st.file_uploader(
    "Upload a crop photo (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"]
)

image_to_process = None
image_name = "sample.jpg"

if uploaded:
    image_to_process = uploaded.getvalue()
    image_name = uploaded.name
    st.image(uploaded, caption="Uploaded Crop Image", use_container_width=True)
elif selected_sample and sample_options[selected_sample]:
    sample_path = Path(sample_options[selected_sample])
    with open(sample_path, "rb") as f:
        image_to_process = f.read()
    image_name = sample_path.name
    st.image(sample_path, caption=f"Selected Sample: {selected_sample}", use_container_width=True)

if image_to_process:
    if st.button("🔍 Run CropSense Intelligence", type="primary"):
        with st.spinner("Executing multi-task inference..."):
            try:
                response = requests.post(
                    API_URL,
                    files={"file": (image_name, image_to_process, "image/jpeg")},
                    timeout=30
                )

                if response.status_code != 200:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    st.stop()

                result = response.json()
                st.divider()

                is_confident = result["status"] == "confident"

                if is_confident:
                    st.success("✅ **High-Confidence Prediction Passed (Confidence Floor ≥ 70%)**")
                else:
                    st.warning("⚠️ **Abstention Triggered / Low-Confidence Prediction (Flagged for Review)**")

                # Metrics row 1: Predictions
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("🌱 Crop Type", result.get("crop") or "Unknown")
                with c2:
                    st.metric("📈 Growth Stage", result.get("stage") or "Unknown")
                with c3:
                    cond_name = result.get("condition") or "Unknown"
                    st.metric("🩺 Health / Disease", cond_name)

                # Metrics row 2: Confidences
                conf_c1, conf_c2, conf_c3 = st.columns(3)
                crop_conf = result.get("crop_confidence")
                stage_conf = result.get("stage_confidence")
                dis_conf = result.get("disease_confidence")

                with conf_c1:
                    if crop_conf is not None:
                        st.metric("Crop Confidence", f"{crop_conf * 100:.1f}%")
                with conf_c2:
                    if stage_conf is not None:
                        st.metric("Stage Confidence", f"{stage_conf * 100:.1f}%")
                with conf_c3:
                    if dis_conf is not None:
                        st.metric("Disease Confidence", f"{dis_conf * 100:.1f}%")

                st.divider()

                # Actionable Disease Advisory Section
                st.markdown("### 📋 Agronomic Advisory & Treatment Recommendation")
                condition = result.get("condition", "Healthy")
                advisory = ADVISORIES.get(condition, {
                    "icon": "ℹ️",
                    "desc": "Inspection required for unknown or atypical foliar symptoms.",
                    "action": "Consult local Krishi Vigyan Kendra (KVK) or agricultural extension specialist."
                })

                st.markdown(f"**Condition Status:** {advisory.get('icon', '')} `{condition}`")
                st.info(f"**Diagnosis Details:** {advisory.get('desc', '')}")
                st.warning(f"**Recommended Action:** {advisory.get('action', '')}")

                st.caption(
                    f"Inference Engine: CropSense Multi-Task EfficientNet-B0 (PyTorch) | Status: `{result.get('status')}`"
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Could not connect to CropSense FastAPI backend (http://127.0.0.1:8000). "
                    "Ensure the API server is started with: `python -m uvicorn inference.router:app --port 8000`"
                )
            except Exception as e:
                st.error(f"Execution Error: {e}")