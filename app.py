import cv2
import numpy as np
import streamlit as st
from PIL import Image
from processor import analyze_leaf

st.set_page_config(page_title="Yaprak Hastalığı Analizi", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 0.5rem !important;
    }
    [data-testid="stAppViewContainer"], html, body {
        scrollbar-width: none;
        -ms-overflow-style: none;
    }
    [data-testid="stAppViewContainer"]::-webkit-scrollbar,
    html::-webkit-scrollbar,
    body::-webkit-scrollbar {
        display: none;
    }
    [data-testid="stMetricValue"] {
    font-size: 2rem !important;
    white-space: normal !important;
    overflow: visible !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Tarımda Görüntü İşleme: Yaprak Hastalığı Analizi")

filter_method = st.sidebar.selectbox("Filtre Türü", ["median", "gaussian"])
filter_kernel = st.sidebar.slider("Filtre Çekirdek Boyutu", 3, 11, 5, step=2)
morph_kernel = st.sidebar.slider("Morfoloji Çekirdek Boyutu", 3, 15, 7, step=2)
hue_low = st.sidebar.slider("Alt Hue (Kahverengi)", 0, 25, 8)
hue_high = st.sidebar.slider("Üst Hue (Sarı)", 20, 45, 34)
min_sat = st.sidebar.slider("Min Doygunluk", 20, 120, 45)

uploaded = st.file_uploader("Yaprak fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])
if uploaded:
    bgr = cv2.cvtColor(np.array(Image.open(uploaded).convert("RGB")), cv2.COLOR_RGB2BGR)
    res = analyze_leaf(bgr, filter_method, filter_kernel, morph_kernel, hue_low, hue_high, min_sat)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Yaprak Alanı", f"{res.total_leaf_pixels:,} px")
    m2.metric("Hastalıklı Alan", f"{res.diseased_pixels:,} px")
    m3.metric("Hastalık Şiddeti", f"%{res.disease_ratio:.2f}")
    m4.metric("Durum", res.severity_label)

    c1, c2, c3, c4 = st.columns(4)
    c1.image(res.filtered_rgb, caption="1. Filtrelenmiş", use_container_width=True)
    c2.image(res.leaf_mask, caption="2. Yaprak Maskesi", use_container_width=True)
    c3.image(res.disease_mask, caption="3. Hastalık Maskesi", use_container_width=True)
    c4.image(res.overlay_rgb, caption="4. Sonuç", use_container_width=True)