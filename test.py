import streamlit as st
import cv2
import numpy as np
from PIL import Image
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# 設定頁面寬度
st.set_page_config(layout="wide", page_title="AOI 影像處理實驗室")

# --- 側邊欄：功能設定 ---
with st.sidebar:
    st.title("🎛️ 控制面板")
    uploaded_file = st.file_uploader("上傳影像 (JPG, PNG, TIF)", type=["jpg", "png", "tif"])
    
    st.divider()
    st.subheader("🛠️ 處理步驟疊加")
    # 使用勾選框，順序固定為先濾波 -> 灰階 -> 邊緣/二值化
    do_blur = st.checkbox("1. 高斯模糊 (去噪)")
    do_gray = st.checkbox("2. 轉為灰階", value=True)
    do_canny = st.checkbox("3. Canny 邊緣檢測")
    do_threshold = st.checkbox("4. 二值化處理")
    
    st.divider()
    st.subheader("📊 數據分析")
    do_hist = st.checkbox("顯示直方圖 (Histogram)")

# --- 主畫面邏輯 ---
st.title("🔬 AOI 影像處理與分析系統")

if uploaded_file is not None:
    # 1. 讀取影像 (OpenCV 格式)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)
    h, w, _ = img_original.shape
    
    # 建立處理副本
    img_processed = img_original.copy()

    # --- 2. 執行影像處理流水線 (Pipeline) ---
    # 步驟 1: 高斯模糊
    if do_blur:
        img_processed = cv2.GaussianBlur(img_processed, (5, 5), 0)
        
    # 步驟 2: 灰階化
    if do_gray:
        if len(img_processed.shape) == 3:
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
            
    # 步驟 3: Canny 邊緣檢測
    if do_canny:
        # 防錯：Canny 必須在灰階下執行
        temp_for_canny = img_processed
        if len(temp_for_canny.shape) == 3:
            temp_for_canny = cv2.cvtColor(temp_for_canny, cv2.COLOR_BGR2GRAY)
        img_processed = cv2.Canny(temp_for_canny, 50, 150)
        
    # 步驟 4: 二值化
    if do_threshold:
        temp_for_thresh = img_processed
        if len(temp_for_thresh.shape) == 3:
            temp_for_thresh = cv2.cvtColor(temp_for_thresh, cv2.COLOR_BGR2GRAY)
        _, img_processed = cv2.threshold(temp_for_thresh, 127, 255, cv2.THRESH_BINARY)

    # --- 3. 顯示區塊 ---
    col_img, col_data = st.columns([2, 1])

    with col_img:
        st.subheader("🖼️ 影像操作區 (可縮放)")
        
        # 建立 Folium 地圖
        # 設定 min_zoom=0 且 max_bounds=True 確保縮小時不見邊界外
        m = folium.Map(
            crs='Simple',
            location=[h/2, w/2],
            zoom_start=1,
            min_zoom=0,
            max_bounds=True,
            min_lat=0,
            max_lat=h,
            min_lon=0,
            max_lon=w,
            tiles=None
        )

        # 疊加處理後的圖片
        folium.raster_layers.ImageOverlay(
            image=img_processed,
            bounds=[[0, 0], [h, w]],
            opacity=1.0
        ).add_to(m)

        # 限制操作範圍
        m.fit_bounds([[0, 0], [h, w]])
        
        # 渲染地圖
        st_folium(m, width="100%", height=600)

    with col_data:
        if do_hist:
            st.subheader("📊 直方圖分析")
            fig, ax = plt.subplots()
            
            # 判斷目前影像是單通道還是多通道來繪製直方圖
            if len(img_processed.shape) == 2:
                hist = cv2.calcHist([img_processed], [0], None, [256], [0, 256])
                ax.plot(hist, color='black')
                ax.set_title("Grayscale Histogram")
            else:
                for i, color in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([img_processed], [i], None, [256], [0, 256])
                    ax.plot(hist, color=color)
                ax.set_title("RGB Color Histogram")
            
            ax.set_xlim([0, 256])
            st.pyplot(fig)
        else:
            st.info("👈 請從左側勾選「顯示直方圖」來分析像素分佈。")
            
else:
    st.warning("請先上傳影像檔案以開始分析。")
