import streamlit as st
import requests

st.title("CropSense")

uploaded = st.file_uploader("Upload a crop photo", type=["jpg", "png"])
if uploaded:
    resp = requests.post("http://localhost:8000/predict", files={"file": uploaded})
    result = resp.json()
    st.image(uploaded)
    st.json(result)
    st.caption(f"Source: {result['source']} • Confidence: {result['confidence']:.2f}")