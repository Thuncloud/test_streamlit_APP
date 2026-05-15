import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_ideal_lowpass(img, d0):
    """
    實作理想低通濾波器 (Ideal Lowpass Filter)
    """
    # 1. 取得影像尺寸並執行傅立葉轉換
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # 進行 FFT 並移至中心
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 2. 建立理想低通遮罩 (Mask)
    # 建立與原圖相同尺寸的黑色畫布 (全為 0)
    mask = np.zeros((rows, cols), np.uint8)
    
    # 計算每個點到中心點 (crow, ccol) 的距離
    # 只有距離小於 d0 的區域設為 1 (白色)
    u = np.linspace(0, rows - 1, rows)
    v = np.linspace(0, cols - 1, cols)
    U, V = np.meshgrid(v, u)
    dist = np.sqrt((U - ccol)**2 + (V - crow)**2)
    
    mask[dist <= d0] = 1
    
    # 3. 執行濾波 (頻譜圖直接與遮罩相乘)
    fshift_filtered = fshift * mask
    
    # 4. 反傅立葉轉換回空間域
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    
    # 取絕對值並正規化回 0-255
    img_back = np.abs(img_back)
    return np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

def apply_butterworth_lowpass(img, d0, n=2):
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2, cols // 2
    
    # 建立頻域網格
    u = np.linspace(0, rows - 1, rows)
    v = np.linspace(0, cols - 1, cols)
    V, U = np.meshgrid(v, u)
    dist = np.sqrt((U - crow)**2 + (V - ccol)**2)
    
    # 巴特沃斯公式：1 / (1 + (D/D0)^(2n))
    # 加上 1e-5 避免除以零
    h = 1 / (1 + (dist / (d0 + 1e-5))**(2 * n))
    
    # 執行濾波 (FFT -> Shift -> Multiply -> IShift -> IFFT)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * h
    
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    return np.uint8(cv2.normalize(np.abs(img_back), None, 0, 255, cv2.NORM_MINMAX))

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
        "Hough 直線偵測",
        "方向性邊緣偵測",
        "圓圈檢測",
        "理想低通濾波器",
        "巴特沃斯低通濾波器"
    ]
    
    selected_steps = st.multiselect(
        "請依序選擇處理步驟：", 
        step_options,
        help="演算法執行的先後順序會極大影響 AOI 檢測結果"
    )
    
    st.divider()
    do_hist = st.checkbox("顯示像素直方圖分析及頻譜分析圖")

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

        elif step == "理想低通濾波器":
            d0 = st.sidebar.slider("截止頻率 (D0)", 1, 200, 50)
    
            # 關鍵：如果是彩色影像，先轉灰階
            if len(img_current.shape) == 3:
                img_input = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
            else:
                img_input = img_current
        
            img_current = apply_ideal_lowpass(img_input, d0)

        elif step == "巴特沃斯低通濾波器":
            d0 = st.sidebar.slider("截止頻率 (D0)", 1, 200, 50)
            n = st.sidebar.slider("巴特沃斯 (n)", 1, 5, 2)
    
            # 關鍵：如果是彩色影像，先轉灰階
            if len(img_current.shape) == 3:
                img_input = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
            else:
                img_input = img_current
        
            img_current = apply_butterworth_lowpass(img_input, d0, n)
            
        elif step == "高斯模糊 (濾波)":
            k = st.sidebar.slider("高斯核大小", 1, 15, 5, step=2)
            sig = st.sidebar.slider("SigmaX", 0.0, 5.0, 0.0)
            
            img_current = cv2.GaussianBlur(img_current, (k, k), sig)
        
        elif step == "中值濾波 (去噪)":
            k = st.sidebar.slider("中值濾波核大小", 3, 15, 5, step=2)
            img_current = cv2.medianBlur(img_current, k)
            
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
            h_thresh = st.sidebar.slider("Hough 投票閾值", 10, 200, 100)
            min_len = st.sidebar.slider("最小線段長度", 1, 200, 50)
            max_gap = st.sidebar.slider("最大線段間隙", 1, 50, 10)
            
            line_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(edges.shape) == 2 else edges.copy()
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, h_thresh, minLineLength=min_len, maxLineGap=max_gap)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img_current = line_img

        elif step == "圓圈檢測":
            # 1. 前處理 (圓檢測對雜訊極敏感，必須平滑化)
            gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
            # 教材重點：使用高斯模糊降噪
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
            # 2. 側邊欄控制
            p2 = st.sidebar.slider("圓心投票閾值 (param2)", 10, 100, 30)
            min_dist = st.sidebar.slider("圓心最小距離", 10, 200, 50)
            r_limit = st.sidebar.slider("半徑範圍", 0, 500, (10, 100))
    
            # 3. 執行偵測
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, 
                minDist=min_dist, param1=100, param2=p2, 
                minRadius=r_limit[0], maxRadius=r_limit[1]
            )
    
            # 4. 繪製結果
            res_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # 畫外圓 (綠色)
                    cv2.circle(res_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # 畫圓心 (紅色)
                    cv2.circle(res_img, (i[0], i[1]), 2, (0, 0, 255), 3)
                    
            img_current = res_img

        elif step == "方向性邊緣偵測":
            # 1. 確保灰階處理
            temp = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
    
            # 2. 側邊欄控制項
            direction = st.sidebar.selectbox("選擇偵測方向", ["水平 (0°)", "垂直 (90°)", "垂直 + 水平", "+45°", "-45°"])
    
            # 這裡加入拉桿參數
            # alpha 分配權重 (0.0 為純水平, 1.0 為純垂直)
            blend_weight = st.sidebar.slider("水平/垂直融合比例", 0.0, 1.0, 0.5)
            # brightness 補償 (有時候過濾完太暗，可以手動調亮)
            brightness_offset = st.sidebar.slider("邊緣亮度補償", 0, 100, 0)
    
            # 3. 定義卷積核 (Kernels)
            h_kernel = np.array([[-1, -1, -1], [ 2,  2,  2], [-1, -1, -1]])
            v_kernel = np.array([[-1,  2, -1], [-1,  2, -1], [-1,  2, -1]])
            p45_kernel = np.array([[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]])
            m45_kernel = np.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]])
    
            # 4. 運算邏輯
            if direction == "垂直 + 水平":
                # 教材重點：使用 CV_16S 避免負值被歸零
                res_h = cv2.filter2D(temp, cv2.CV_16S, h_kernel)
                res_v = cv2.filter2D(temp, cv2.CV_16S, v_kernel)
                
                abs_h = cv2.convertScaleAbs(res_h)
                abs_v = cv2.convertScaleAbs(res_v)
        
                # 利用拉桿調整混合比例：dst = src1*alpha + src2*beta + gamma
                # 我們讓 alpha = 1 - blend_weight, beta = blend_weight
                img_current = cv2.addWeighted(abs_h, 1 - blend_weight, abs_v, blend_weight, brightness_offset)
        
            else:
                # 單一方向偵測
                if direction == "水平 (0°)": curr_kernel = h_kernel
                elif direction == "垂直 (90°)": curr_kernel = v_kernel
                elif direction == "+45°": curr_kernel = p45_kernel
                elif direction == "-45°": curr_kernel = m45_kernel
        
                # 執行濾波並補償亮度
                filtered = cv2.filter2D(temp, cv2.CV_16S, curr_kernel)
                img_current = cv2.convertScaleAbs(filtered + brightness_offset)
            
    

    # 3. 顯示區塊
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🖼️ 影像對比結果")
        if selected_steps:
            st.caption(f"執行路徑: 原圖 ➔ {' ➔ '.join(selected_steps)}")
        
        # 建立內部分欄：左邊放原圖，右邊放處理後的圖
        sub_col_orig, sub_col_proc = st.columns(2)
        
        with sub_col_orig:
            # 準備原圖 (需轉為 RGB 顏色才正確)
            img_orig_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            st.image(img_orig_rgb, caption="原始影像 (Original)", use_container_width=True)
            
        with sub_col_proc:
            # 準備處理後的影像
            if len(img_current.shape) == 3:
                display_img = cv2.cvtColor(img_current, cv2.COLOR_BGR2RGB)
            else:
                display_img = img_current
            st.image(display_img, caption="處理後影像 (Processed)", use_container_width=True)

    with col_right:
        if do_hist:
            st.subheader("📊 像素直方圖")
            # 注意：這裡顯示的是「處理後影像」的直方圖，這對觀察濾波效果很有幫助
            fig, ax = plt.subplots()
            if len(img_current.shape) == 2:
                # 繪製灰階直方圖 (使用教材提到的 ravel() 扁平化處理)
                ax.hist(img_current.ravel(), 256, [0, 256], color='black')
                ax.set_title("Grayscale Histogram")
            else:
                # 繪製彩色直方圖
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([img_current], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col)
                ax.set_title("RGB Histogram")
            
            ax.set_xlim([0, 256])
            st.pyplot(fig)
            
            st.subheader("🌌 頻域分析 (FFT)")
            # 1. 確保影像是灰階才能做 FFT
            if len(img_current.shape) == 3:
                fft_input = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
            else:
                fft_input = img_current
        
            # 2. 執行傅立葉轉換
            f = np.fft.fft2(fft_input)
            fshift = np.fft.fftshift(f) # 將低頻位移至中心
            
            # 3. 計算振幅譜 (Magnitude Spectrum)
            # 取絕對值後加上 log 轉換，否則中心點太亮會看不到細節
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
            # 4. 顯示圖表
            fig_fft, ax_fft = plt.subplots()
            ax_fft.imshow(magnitude_spectrum, cmap='gray')
            ax_fft.set_title('Frequency Spectrum')
            ax_fft.axis('off') # 隱藏座標軸比較美觀
            st.pyplot(fig_fft)
        else:
            st.info("👈 勾選側邊欄可查看處理後的數據分佈。")

else:
    st.warning("請先上傳圖片以開始 AOI 演算法測試。")
