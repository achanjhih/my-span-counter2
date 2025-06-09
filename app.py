
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Span & Pillar Counter", layout="wide")
st.title("📐 スパン数 & 支柱数 検出アプリ")

uploaded_file = st.file_uploader("図面PDFをアップロード", type=["pdf", "png", "jpg", "jpeg"])
if uploaded_file:
    st.info("画像処理中...")
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        if uploaded_file.name.endswith(".pdf"):
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(uploaded_file.read())
            images[0].save(tmp.name, format='PNG')
        else:
            img = Image.open(uploaded_file).convert("RGB")
            img.save(tmp.name)

        image = cv2.imread(tmp.name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        span_boxes = []
        pillar_circles = []

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if 30 < w < 300 and 30 < h < 300:
                    span_boxes.append((x, y, w, h))
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                   param1=50, param2=15, minRadius=3, maxRadius=15)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)
                pillar_circles.append(i)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"🟥 スパン: {len(span_boxes)}個　🔵 支柱: {len(pillar_circles)}本", use_column_width=True)
