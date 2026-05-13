import streamlit as st
import cv2
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# 設定頁面
st.set_page_config(layout="wide", page_title="AOI 影像處理實驗室")

# --- 側邊欄控制面板 ---
with st.sidebar:
    st.title("🎛️ 控制面板")
    uploaded_file = st.file_uploader("上傳影像 (JPG, PNG, TIF)", type=["jpg", "png", "tif"])
    
    st.divider()
    st.subheader("🧪 處理流程設計")
    # 使用 multiselect 來決定「順序」
    step_options = ["高斯模糊", "中值濾波", "轉為灰階", "二值化", "Canny 邊緣檢測"]
    selected_steps = st.multiselect(
        "請依序選擇處理步驟：", 
        step_options,
        help="選擇的先後順序將直接影響影像處理的結果"
    )
    
    st.divider()
    st.subheader("📊 數據分析")
    do_hist = st.checkbox("顯示直方圖 (Histogram)")

# --- 主畫面邏輯 ---
st.title("🔬 AOI 影像動態處理系統")

if uploaded_file is not None:
    # 1. 讀取影像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)
    h, w, _ = img_original.shape
    
    # 建立承接變數
    img_current = img_original.copy()

    # 2. 執行動態流水線 (依照選取順序)
    for step in selected_steps:
        if step == "高斯模糊":
            img_current = cv2.GaussianBlur(img_current, (5, 5), 0)
        
        elif step == "中值濾波":
            img_current = cv2.medianBlur(img_current, 5)
            
        elif step == "轉為灰階":
            if len(img_current.shape) == 3:
                img_current = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
                
        elif step == "二值化":
            # 防錯：若目前是彩色，內部先轉灰階
            temp = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
            _, img_current = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
            
        elif step == "Canny 邊緣檢測":
            temp = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
            img_current = cv2.Canny(temp, 50, 150)

    # 3. 佈局：左邊地圖，右邊數據
    col_img, col_data = st.columns([2, 1])

    with col_img:
        if selected_steps:
            st.info(f"當前路徑：原圖 ➔ {' ➔ '.join(selected_steps)}")
        
        # 建立 Folium 地圖 (Simple CRS)
        m = folium.Map(
            crs='Simple',
            location=[h/2, w/2],
            zoom_start=0,
            min_zoom=-2,
            max_bounds=True,
            min_lat=0, max_lat=h,
            min_lon=0, max_lon=w,
            tiles=None
        )

        # 疊加影像
        folium.raster_layers.ImageOverlay(
            image=img_current,
            bounds=[[0, 0], [h, w]],
            opacity=1.0
        ).add_to(m)

        # 強制初始畫面完整顯示
        m.fit_bounds([[0, 0], [h, w]])
        
        st_folium(m, width="100%", height=600, key="aoi_map")

    with col_data:
        if do_hist:
            st.subheader("📊 直方圖分析")
            fig, ax = plt.subplots()
            
            if len(img_current.shape) == 2: # 灰階/二值/Canny
                hist = cv2.calcHist([img_current], [0], None, [256], [0, 256])
                ax.plot(hist, color='black')
            else: # 彩色
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([img_current], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col)
            
            ax.set_xlim([0, 256])
            st.pyplot(fig)
        else:
            st.info("👈 請從左側勾選「顯示直方圖」來分析。")

else:
    st.warning("請先上傳影像檔案。")
