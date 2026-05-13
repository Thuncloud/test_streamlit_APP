import streamlit as st
import cv2
import numpy as np
from streamlit_folium import st_folium
import folium
from PIL import Image

uploaded_file = st.sidebar.file_uploader("上傳影像", type=["jpg", "png", "tif"])

if uploaded_file is not None:
    # 1. 這裡先建立 img_original
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)

    # 2. 接下來才能進行 copy() 或是其他處理
    img = img_original.copy()
    
    # 3. 執行你的動態步驟 (Pipeline)
    for step in selected_steps:
        if step == "灰階化":
            # 檢查是否已經是灰階，避免重複轉型報錯
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        elif step == "高斯模糊":
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
        elif step == "二值化":
            if len(img.shape) == 3: # 如果還是彩色，先強制轉灰階
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
        elif step == "Canny 邊緣檢測":
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Canny(img, 50, 150)
            
        elif step == "中值濾波":
            img = cv2.medianBlur(img, 5)

    # 3. 顯示最終疊加結果
    st.image(img, caption=f"經過步驟：{' -> '.join(selected_steps) if selected_steps else '原始影像'}")
