import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="CropSense",
    page_icon="🌱",
    layout="centered"
)


st.title("🌱 CropSense")
st.write(
    "AI-assisted crop analysis with confidence-aware prediction."
)


uploaded = st.file_uploader(
    "Upload a crop photo",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded:

    st.image(
        uploaded,
        caption="Uploaded crop image",
        use_container_width=True
    )

    if st.button("Analyze Crop", type="primary"):

        with st.spinner("Analyzing image..."):

            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type
                        )
                    },
                    timeout=30
                )

                if response.status_code != 200:
                    st.error(
                        f"API error: {response.text}"
                    )
                    st.stop()

                result = response.json()

                st.divider()

                if result["status"] == "confident":

                    st.success(
                        "Confident prediction"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Crop",
                            result["crop"]
                        )

                        st.metric(
                            "Growth Stage",
                            result["stage"]
                        )

                    with col2:
                        st.metric(
                            "Condition",
                            result["condition"]
                        )

                        st.metric(
                            "Confidence",
                            f"{result['confidence'] * 100:.1f}%"
                        )

                    st.caption(
                        f"Prediction source: {result['source']}"
                    )

                else:

                    st.warning(
                        "The model is not confident enough "
                        "to provide a reliable prediction."
                    )

                    st.metric(
                        "Confidence",
                        f"{result['confidence'] * 100:.1f}%"
                    )

                    st.info(
                        "This image should be reviewed by "
                        "the teacher model or a human expert."
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the CropSense API. "
                    "Make sure the FastAPI server is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The CropSense API took too long to respond."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )