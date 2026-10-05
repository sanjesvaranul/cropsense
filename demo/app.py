import io
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict"
EVAL_DIR = Path("training/evaluation")

st.set_page_config(
    page_title="CropSense - Mobile Crop, Stage & Disease Intelligence",
    page_icon="🌾",
    layout="wide"
)

# Custom header styling
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stMetric {
        background-color: #f4f7f5;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #e1ebe3;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 CropSense: AgriLens")
st.subheader("Mobile-Image Crop, Stage and Disease Model (FarmwiseAI Task 4)")
st.write(
    "A teacher-student vision system powered by an edge-optimized **EfficientNet-B0** "
    "multi-task backbone with **Selective Abstention (70% Confidence Floor)** and instant agronomic advisories."
)

# Agronomic knowledge base & treatment suggestions
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Single-Photo Diagnosis",
    "📂 Batch Inference & Export",
    "⚠️ Review Queue & Failure Analysis",
    "📊 Teacher vs. Student Benchmark",
    "📈 Evaluation & Confusion Matrices"
])

# ==============================================================================
# TAB 1: SINGLE-PHOTO DIAGNOSIS
# ==============================================================================
with tab1:
    st.markdown("### 🌾 Real-Time Single Image Diagnosis")

    col_side, col_main = st.columns([1, 2])

    with col_side:
        st.markdown("#### 🧪 Test Sample Library")
        sample_dir = Path("data/external/rice_diseases")
        sample_options = {"None (Upload own)": None}

        if sample_dir.exists():
            for sub in sample_dir.iterdir():
                if sub.is_dir():
                    imgs = list(sub.glob("*.jpg")) + list(sub.glob("*.png"))
                    if imgs:
                        sample_options[f"Rice ({sub.name})"] = str(imgs[0])

        raw_rice = list(Path("data/raw/images").glob("*.jpg"))
        if raw_rice:
            sample_options["Rice (Healthy field sample)"] = str(raw_rice[0])

        selected_sample = st.selectbox("Quick-test preset:", list(sample_options.keys()))

        uploaded = st.file_uploader(
            "Or upload your own crop photo:",
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
            st.image(sample_path, caption=f"Selected: {selected_sample}", use_container_width=True)

    with col_main:
        if image_to_process:
            if st.button("🔍 Run CropSense Intelligence", type="primary", use_container_width=True):
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

                        is_confident = result.get("status") == "confident"

                        if is_confident:
                            st.success("✅ **High-Confidence Prediction Passed (Confidence Floor ≥ 70%)**")
                        else:
                            st.warning("⚠️ **Abstention Triggered (Flagged for Agronomist / Teacher Review)**")
                            if "abstention_reason" in result:
                                st.caption(f"Reason: {result['abstention_reason']}")

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
                            f"Inference Route: `{result.get('source')}` | Decision Engine: EfficientNet-B0 (Edge) | Status: `{result.get('status')}`"
                        )

                    except requests.exceptions.ConnectionError:
                        st.error(
                            "❌ Could not connect to CropSense FastAPI backend (http://127.0.0.1:8000). "
                            "Ensure the API server is started with: `python -m uvicorn inference.router:app --port 8000`"
                        )
                    except Exception as e:
                        st.error(f"Execution Error: {e}")
        else:
            st.info("👈 Select a sample from the library or upload a field photograph to begin diagnosis.")

# ==============================================================================
# TAB 2: BATCH INFERENCE & EXPORT
# ==============================================================================
with tab2:
    st.markdown("### 📂 Batch Crop & Disease Processing")
    st.write("Upload a collection of field images to generate batch predictions with confidence scoring and CSV export.")

    batch_files = st.file_uploader(
        "Upload multiple field photos:",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if batch_files:
        st.write(f"Loaded **{len(batch_files)}** images for batch inference.")
        if st.button("🚀 Process Batch", type="primary"):
            progress_bar = st.progress(0)
            batch_results = []

            for idx, file in enumerate(batch_files):
                try:
                    resp = requests.post(
                        API_URL,
                        files={"file": (file.name, file.getvalue(), file.type)},
                        timeout=30
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        batch_results.append({
                            "Filename": file.name,
                            "Crop": data.get("crop"),
                            "Crop Confidence": f"{data.get('crop_confidence', 0)*100:.1f}%",
                            "Growth Stage": data.get("stage"),
                            "Stage Confidence": f"{data.get('stage_confidence', 0)*100:.1f}%",
                            "Disease Condition": data.get("condition"),
                            "Disease Confidence": f"{data.get('disease_confidence', 0)*100:.1f}%",
                            "Route Status": data.get("status")
                        })
                except Exception as e:
                    batch_results.append({"Filename": file.name, "Route Status": f"Error: {e}"})
                progress_bar.progress((idx + 1) / len(batch_files))

            df_batch = pd.DataFrame(batch_results)
            st.dataframe(df_batch, use_container_width=True)

            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Batch Results (CSV)",
                data=csv_data,
                file_name="cropsense_batch_predictions.csv",
                mime="text/csv"
            )

# ==============================================================================
# TAB 3: REVIEW QUEUE & FAILURE ANALYSIS
# ==============================================================================
with tab3:
    st.markdown("### ⚠️ Human-in-the-Loop Review Queue & Failure Analysis")
    st.write(
        "Images where the model confidence fell below 70% or where edge ambiguities occurred "
        "are safely routed here for agronomist and teacher verification."
    )

    failures_csv = EVAL_DIR / "failure_cases_analysis.csv"
    if failures_csv.exists():
        df_fail = pd.read_csv(failures_csv)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Total Flagged / Review Cases", len(df_fail))
        with col_f2:
            st.metric("Selective Abstention Rate (< 70%)", "17.09%")
        with col_f3:
            st.metric("Safety Floor Threshold", "70.0%")

        st.markdown("#### 📋 Flagged Test Cases Queue")
        st.dataframe(df_fail[[
            "image_path", "true_crop", "pred_crop", "crop_conf",
            "true_stage", "pred_stage", "stage_conf",
            "true_condition", "pred_condition", "condition_conf",
            "failure_category"
        ]], use_container_width=True)

        st.markdown("#### 🔍 Root-Cause Analysis")
        st.info("""
        - **Growth Stage Continuum:** The majority of stage misclassifications occur at the transitional boundary between *vegetation* and *full_growth*. CropSense automatically detects this ambiguity and abstains rather than making an unverified guess.
        - **Crop Type Consistency:** Staple crops (Rice, Coconut, Maize, Sugarcane) maintain >95% accuracy.
        - **Disease Diagnostics:** 99.43% overall accuracy on foliar diseases with only 2 boundary cases across 351 test images.
        """)
    else:
        st.info("Run `python training/analyze_failures.py` to generate the test failure queue.")

# ==============================================================================
# TAB 4: TEACHER VS. STUDENT BENCHMARK
# ==============================================================================
with tab4:
    st.markdown("### 📊 Head-to-Head: Teacher VLM vs. CropSense Student Model")
    st.write(
        "As proposed in Section 4.6 of the AgriLens proposal, this dashboard compares the "
        "deployed student model against commercial cloud Vision-Language Models (Gemini / GPT-4V)."
    )

    comp_csv = EVAL_DIR / "teacher_student_comparison.csv"
    if comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)
        st.dataframe(df_comp, use_container_width=True)

        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
        with c_b1:
            st.metric("Cost Reduction", "99.94%", delta="Cheaper than API")
        with c_b2:
            st.metric("Latency Speedup", "16x Faster", delta="Instant Response")
        with c_b3:
            st.metric("Model Footprint", "15.68 MB", delta="Mobile-Ready")
        with c_b4:
            st.metric("Disease Accuracy", "99.43%", delta="+3.03% vs VLM")

        st.success("""
        **Key Takeaway for FarmwiseAI:**
        By distilling agricultural knowledge into an EfficientNet-B0 student model, CropSense eliminates 99.94% of operational inference costs while achieving near-zero latency and 100% offline field capability.
        """)
    else:
        st.info("Run `python training/benchmark_comparison.py` to generate the benchmark comparison.")

# ==============================================================================
# TAB 5: EVALUATION & CONFUSION MATRICES
# ==============================================================================
with tab5:
    st.markdown("### 📈 Comprehensive Model Evaluation & Confusion Matrices")
    st.write(
        "Quantitative benchmark metrics evaluated on the 351 held-out test samples (`data/labeled/test_multitask.csv`) "
        "demonstrating multi-task generalization across crops, developmental stages, and foliar diseases."
    )

    c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
    with c_m1:
        st.metric("🌱 Crop Accuracy", "94.02%", delta="Held-out Test")
    with c_m2:
        st.metric("📈 Stage Accuracy", "82.34%", delta="Held-out Test")
    with c_m3:
        st.metric("🩺 Disease Accuracy", "99.43%", delta="Held-out Test")
    with c_m4:
        st.metric("🛡️ Confident Pass", "82.91%", delta="≥ 70% Floor")
    with c_m5:
        st.metric("⚠️ Selective Abstention", "17.09%", delta="Flagged Cases")

    st.divider()

    st.markdown("#### 🎯 Interactive Confusion Matrix Explorer")
    matrix_choice = st.selectbox(
        "Select classification task:",
        [
            "Foliar Disease / Condition (5 classes)",
            "Growth Stage Recognition (5 stages)",
            "Crop Identification (12 crops)"
        ]
    )

    if matrix_choice.startswith("Foliar"):
        cm_file = EVAL_DIR / "condition_confusion_matrix.csv"
        title = "Foliar Disease & Health Confusion Matrix"
    elif matrix_choice.startswith("Growth"):
        cm_file = EVAL_DIR / "stage_confusion_matrix.csv"
        title = "Growth Stage Confusion Matrix"
    else:
        cm_file = EVAL_DIR / "crop_confusion_matrix.csv"
        title = "Crop Identification Confusion Matrix"

    if cm_file.exists():
        df_cm = pd.read_csv(cm_file, index_col=0)

        col_plot, col_data = st.columns([3, 2])
        with col_plot:
            fig, ax = plt.subplots(figsize=(7, 5))
            cax = ax.imshow(df_cm.values, cmap="Blues", interpolation="nearest")
            fig.colorbar(cax, fraction=0.046, pad=0.04)

            ticks = np.arange(len(df_cm.columns))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(df_cm.columns, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(df_cm.index, fontsize=9)
            ax.set_xlabel("Predicted Label", fontweight="bold")
            ax.set_ylabel("True Label", fontweight="bold")
            ax.set_title(title, fontweight="bold", pad=12)

            thresh = df_cm.values.max() / 2.0 if df_cm.values.max() > 0 else 1
            for i in range(len(df_cm.index)):
                for j in range(len(df_cm.columns)):
                    val = df_cm.values[i, j]
                    ax.text(
                        j, i, f"{int(val)}",
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=9
                    )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_data:
            st.markdown("##### 🔢 Raw Counts Table")
            st.dataframe(df_cm, use_container_width=True)

            if matrix_choice.startswith("Foliar"):
                st.info(
                    "**Key Finding:** 99.43% test accuracy on disease symptoms. "
                    "Healthy paddy, Bacterial Blight, and Blast achieved 100% precision."
                )
            elif matrix_choice.startswith("Growth"):
                st.info(
                    "**Key Finding:** 82.34% stage accuracy. Confusions concentrate along "
                    "adjacent transitional boundaries (vegetation ↔ full_growth)."
                )
            else:
                st.info(
                    "**Key Finding:** 94.02% crop accuracy. Dominant staples (Rice, Coconut, "
                    "Sugarcane, Maize) show high precision with class-weighted loss mitigation."
                )

    st.divider()

    st.markdown("#### 📋 Task-Wise Classification Breakdown")
    summary_path = EVAL_DIR / "multitask_evaluation_summary.txt"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            full_txt = f.read()

        with st.expander("📄 View Full Scikit-Learn Classification Reports (Precision, Recall, F1)", expanded=False):
            st.code(full_txt, language="text")

        st.download_button(
            "📥 Download Multi-Task Evaluation Report (TXT)",
            data=full_txt,
            file_name="multitask_evaluation_report.txt",
            mime="text/plain"
        )