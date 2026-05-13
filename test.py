import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 設定頁面寬度與標題
st.set_page_config(layout="wide", page_title="AOI 影像處理演算法實驗室")

# --- 側邊欄：動態流水線設定 ---
with st.sidebar:
    st.title("🎛️ 演算法控制面板")
    uploaded_file = st.file_uploader("上傳實驗影像", type=["jpg", "png", "tif"])
    
    st.divider()
    st.subheader("🧪 處理流程 (可自訂順序)")
    
    # 根據教材內容整理的演算法清單
    step_options = [
        "轉為灰階", 
        "顏色翻轉",
        "高斯模糊 (濾波)", 
        "中值濾波 (去噪)", 
        "Canny 邊緣檢測", 
        "二值化處理",
        "Hough 直線偵測"
    ]
    
    selected_steps = st.multiselect(
        "請依序選擇處理步驟：", 
        step_options,
        help="演算法執行的先後順序會極大影響 AOI 檢測結果"
    )
    
    st.divider()
    do_hist = st.checkbox("顯示像素直方圖分析")

# --- 主畫面邏輯 ---
st.title("🔬 自動化光學檢測 (AOI) 演算法平台")

if uploaded_file is not None:
    # 1. 影像讀取
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)
    
    # 半成品影像承接變數
    img_current = img_original.copy()

    # 2. 依照教材邏輯執行動態流水線
    for step in selected_steps:
        if step == "Canny 邊緣檢測":
            # 自動轉灰階防錯
            temp = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
            t1 = st.sidebar.slider("Canny 低閾值", 0, 255, 50)
            t2 = st.sidebar.slider("Canny 高閾值", 0, 255, 150)
            img_current = cv2.Canny(temp, t1, t2)
        elif step == "顏色翻轉":
            img_current = cv2.bitwise_not(img_current)
            
        elif step == "高斯模糊 (濾波)":
            img_current = cv2.GaussianBlur(img_current, (5, 5), 0)
        
        elif step == "中值濾波 (去噪)":
            img_current = cv2.medianBlur(img_current, 5)
            
        elif step == "轉為灰階":
            if len(img_current.shape) == 3:
                img_current = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
        
        elif step == "二值化處理":
            temp = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
            thresh = st.sidebar.slider(
                    "thresh:", 
                    min_value=0, 
                    max_value=255, 
                    value=127, 
                    step=1
                )
            _, img_current = cv2.threshold(temp, thresh, 255, cv2.THRESH_BINARY)

        elif step == "Hough 直線偵測":
            # 教材重點：Hough 前通常需要先做邊緣偵測 (Canny)
            # 如果目前影像不是二值圖或邊緣圖，先內部處理
            if len(img_current.shape) == 3:
                gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
            else:
                edges = img_current
            
            # 建立彩色底圖畫線
            line_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(edges.shape) == 2 else edges.copy()
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img_current = line_img

    # 3. 顯示區塊
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🖼️ 影像處理結果")
        if selected_steps:
            st.caption(f"執行路徑: 原圖 ➔ {' ➔ '.join(selected_steps)}")
        
        # 轉換色彩格式供 Streamlit 顯示
        if len(img_current.shape) == 3:
            display_img = cv2.cvtColor(img_current, cv2.COLOR_BGR2RGB)
        else:
            display_img = img_current
            
        st.image(display_img, use_container_width=True)

    with col_right:
        if do_hist:
            st.subheader("📊 像素直方圖")
            fig, ax = plt.subplots()
            if len(img_current.shape) == 2:
                ax.hist(img_current.ravel(), 256, [0, 256], color='black')
            else:
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([img_current], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col)
            st.pyplot(fig)
        else:
            st.info("👈 勾選側邊欄可查看影像數值分佈。")

else:
    st.warning("請先上傳圖片以開始 AOI 演算法測試。")
