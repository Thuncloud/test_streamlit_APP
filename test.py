import streamlit as st
import cv2
import numpy as np
from streamlit_folium import st_folium
import folium
from PIL import Image

step_options = ["灰階化", "高斯模糊", "二值化", "Canny 邊緣檢測", "中值濾波"]
selected_steps = st.sidebar.multiselect("請依序選擇處理步驟：", step_options)
uploaded_file = st.sidebar.file_uploader("上傳影像", type=["jpg", "png", "tif"])

if uploaded_file is not None:
    # 讀取原始影像並備份一份作為處理起點
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)
    
    # 建立處理副本
    img_processed = img_original.copy()

    # --- 1. 在側邊欄建立勾選選項 ---
    with st.sidebar:
        st.header("🛠️ 影像處理步驟")
        
        # 使用勾選框決定是否執行該步驟
        do_gray = st.checkbox("1. 轉為灰階", value=True)
        do_blur = st.checkbox("2. 高斯模糊 (濾噪)")
        do_canny = st.checkbox("3. Canny 邊緣檢測")
        do_hist = st.checkbox("4. 顯示直方圖分析")

    # --- 2. 依照順序執行被勾選的步驟 ---
    
    # 步驟 1: 灰階
    if do_gray:
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
        
    # 步驟 2: 模糊
    if do_blur:
        # 這裡加個小判斷，確保參數是奇數
        img_processed = cv2.GaussianBlur(img_processed, (5, 5), 0)
        
    # 步驟 3: Canny
    if do_canny:
        # Canny 需要單通道影像，如果前面沒勾灰階，這裡幫他轉一下
        if len(img_processed.shape) == 3:
            temp_img = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
        else:
            temp_img = img_processed
        img_processed = cv2.Canny(temp_img, 100, 200)

    # --- 3. 顯示結果 ---
    st.subheader("🖼️ 處理結果")
    
    # 注意：Streamlit 顯示彩色圖需要 RGB 格式
    display_img = img_processed
    if len(img_processed.shape) == 3:
        display_img = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
        
    st.image(display_img, use_container_width=True)

    # 步驟 4: 直方圖 (這屬於數據分析，不影響影像變數)
    if do_hist:
        import matplotlib.pyplot as plt
        st.divider()
        st.subheader("📊 直方圖分析")
        fig, ax = plt.subplots()
        
        if len(img_processed.shape) == 2:
            hist = cv2.calcHist([img_processed], [0], None, [256], [0, 256])
            ax.plot(hist, color='black')
        else:
            for i, col in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([img_processed], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)
        st.pyplot(fig)
