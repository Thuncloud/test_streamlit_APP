import streamlit as st
import cv2
import numpy as np
from streamlit_folium import st_folium
import folium
from PIL import Image

# 1. 在側標欄定義可選的處理步驟
step_options = ["灰階化", "高斯模糊", "二值化", "Canny 邊緣檢測", "中值濾波"]
selected_steps = st.sidebar.multiselect("請依序選擇處理步驟：", step_options)

if uploaded_file is not None:
    # 讀取原始影像
    img = img_original.copy()
    
    # 2. 依照使用者選擇的「順序」跑迴圈
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
