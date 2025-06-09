
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Span & Pillar Counter", layout="centered")
st.title("📐 スパン数 & 支柱数 検出アプリ（画像限定）")

uploaded_file = st.file_uploader("図面画像をアップロードしてください（PNG or JPG）", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.info("画像を処理しています...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img = Image.open(uploaded_file).convert("RGB")
        img.save(tmp.name)
        image = cv2.imread(tmp.name)

        # スパン検出（矩形検出）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        span_boxes = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if 30 < w < 300 and 30 < h < 300:
                    span_boxes.append((x, y, w, h))
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 支柱（青丸）検出
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                   param1=50, param2=15, minRadius=3, maxRadius=15)
        pillar_circles = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)
                pillar_circles.append(i)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="検出結果", use_column_width=True)

        # 集計結果表示
        st.markdown("### ✅ 検出結果")
        st.markdown(f"- 🟥 スパン数：**{len(span_boxes)}** 個")
        st.markdown(f"- 🔵 支柱数　：**{len(pillar_circles)}** 本")
