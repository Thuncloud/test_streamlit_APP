import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 設定頁面
st.set_page_config(layout="wide", page_title="AOI 影像處理實驗室")

# --- 側邊欄控制面板 ---
with st.sidebar:
    st.title("🎛️ 控制面板")
    uploaded_file = st.file_uploader("上傳影像 (JPG, PNG, TIF)", type=["jpg", "png", "tif"])
    
    st.divider()
    st.subheader("🧪 處理流程設計")
    step_options = ["高斯模糊", "中值濾波", "轉為灰階", "二值化", "Canny 邊緣檢測"]
    selected_steps = st.multiselect(
        "請依序選擇處理步驟：", 
        step_options,
        help="選擇的順序將決定處理的順序"
    )
    
    st.divider()
    st.subheader("📊 數據分析")
    do_hist = st.checkbox("顯示直方圖 (Histogram)")

# --- 主畫面邏輯 ---
st.title("🔬 AOI 影像處理系統 (純淨版)")

if uploaded_file is not None:
    # 1. 讀取影像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)
    
    # 建立承接變數
    img_current = img_original.copy()

    # 2. 執行動態流水線
    for step in selected_steps:
        if step == "高斯模糊":
            img_current = cv2.GaussianBlur(img_current, (5, 5), 0)
        elif step == "中值濾波":
            img_current = cv2.medianBlur(img_current, 5)
        elif step == "轉為灰階":
            if len(img_current.shape) == 3:
                img_current = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
        elif step == "二值化":
            temp = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
            _, img_current = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
        elif step == "Canny 邊緣檢測":
            temp = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
            img_current = cv2.Canny(temp, 50, 150)

    # 3. 顯示結果
    col_img, col_data = st.columns([2, 1])

    with col_img:
        st.subheader("🖼️ 影像顯示區")
        if selected_steps:
            st.info(f"當前路徑：原圖 ➔ {' ➔ '.join(selected_steps)}")
        
        # --- 關鍵修改：直接顯示圖片，取消拖移與縮放地圖 ---
        # 如果是彩色圖，轉回 RGB 讓 Streamlit 正確顯示顏色
        if len(img_current.shape) == 3:
            display_img = cv2.cvtColor(img_current, cv2.COLOR_BGR2RGB)
        else:
            display_img = img_current
            
        st.image(display_img, use_container_width=True, caption="處理後的影像")

    with col_data:
        if do_hist:
            st.subheader("📊 直方圖分析")
            fig, ax = plt.subplots()
            if len(img_current.shape) == 2:
                hist = cv2.calcHist([img_current], [0], None, [256], [0, 256])
                ax.plot(hist, color='black')
            else:
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([img_current], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col)
            ax.set_xlim([0, 256])
            st.pyplot(fig)
        else:
            st.info("👈 請勾選側邊欄的直方圖。")

else:
    st.warning("請先上傳影像檔案。")
