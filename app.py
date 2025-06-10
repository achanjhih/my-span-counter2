
import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="スパン・支柱カウンター", layout="centered")
st.title("🧱 スパン・支柱カウンター（画像認識ベータ）")

uploaded_file = st.file_uploader("画像ファイルをアップロードしてください（PNGまたはJPEG）", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 画像を OpenCV 形式に変換
    img = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 青色の支柱（青丸）をHSVで検出
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    post_count = len(contours_blue)

    # スパン（四角形）検出：面積フィルタを使って輪郭抽出
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours_span, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    span_count = sum(1 for cnt in contours_span if cv2.contourArea(cnt) > 10000)

    # 結果表示
    st.subheader("🔍 検出結果")
    st.write(f"✅ スパン数: **{span_count}** 個")
    st.write(f"✅ 支柱数（青丸）: **{post_count}** 本")
