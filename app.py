
import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="スパン＆支柱カウンター（PNG特化）")

st.title("PNG図面からスパン数と支柱数を自動カウント")

uploaded_file = st.file_uploader("PNG画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_container_width=True)

    # NumPy配列に変換
    img_np = np.array(image)

    # スパン色（例：縦1829×横1219のスパン: RGB(2,255,255)）
    span_color = np.array([2, 255, 255])
    span_mask = np.all(img_np == span_color, axis=-1)
    span_count = cv2.connectedComponents(span_mask.astype(np.uint8))[0] - 1

    # 支柱色（例：RGB(0, 0, 255)の青点）
    support_color = np.array([0, 0, 255])
    support_mask = np.all(img_np == support_color, axis=-1)
    support_count = cv2.connectedComponents(support_mask.astype(np.uint8))[0] - 1

    st.subheader("検出結果")
    st.write(f"🟨 スパン数： {span_count} 個")
    st.write(f"🔵 支柱数： {support_count} 本")
