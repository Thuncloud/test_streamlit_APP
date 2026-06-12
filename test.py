import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心演算法工具函式 (頻域處理與濾波)
# ==========================================

def get_fft_shift(img):
    """
    執行快速傅立葉轉換 (FFT) 並將低頻成分位移至中心。
    img: 輸入的灰階影像
    """
    f = np.fft.fft2(img)
    return np.fft.fftshift(f)

def inverse_fft(fshift):
    """
    執行反傅立葉轉換 (IFFT) 將頻域訊號轉回空間域影像。
    fshift: 經過濾波遮罩處理後的頻域矩陣
    """
    f_ishift = np.fft.ifftshift(fshift) # 將中心點移回角落
    img_back = np.fft.ifft2(f_ishift)   # 執行反轉換
    # 取絕對值恢復實數影像，並正規化到 0-255 範圍，最後轉為 uint8 格式
    return np.uint8(cv2.normalize(np.abs(img_back), None, 0, 255, cv2.NORM_MINMAX))

def create_mask(rows, cols, mode="low", filter_type="ideal", d0=30, d1=60, n=2):
    """
    建立頻域濾波遮罩 (Mask)。
    rows, cols  : 影像的尺寸
    mode        : 濾波模式 ("low"低通, "high"高通, "bandpass"帶通, "bandreject"帶拒)
    filter_type : 濾波器數學模型 ("ideal"理想, "gaussian"高斯, "butterworth"巴特沃斯)
    d0, d1      : 截止頻率 (d0 為基准，d1 為帶通/帶拒的高階截止)
    n           : 巴特沃斯濾波器的階數
    """
    crow, ccol = rows // 2, cols // 2
    # 建立網格座標系，計算各點到中心點(低頻)的物理距離
    u = np.linspace(0, rows - 1, rows)
    v = np.linspace(0, cols - 1, cols)
    V, U = np.meshgrid(v, u)
    dist = np.sqrt((U - crow)**2 + (V - ccol)**2)
    dist = np.where(dist == 0, 1e-5, dist) # 避免除以零導致程式崩潰

    # 根據選定類型，計算基本的低通濾波器響應 H(u,v)
    if filter_type == "ideal":
        h = np.zeros((rows, cols))
        h[dist <= d0] = 1
    elif filter_type == "gaussian":
        h = np.exp(-(dist**2) / (2 * (d0**2)))
    elif filter_type == "butterworth":
        h = 1 / (1 + (dist / d0)**(2 * n))

    # 根據 mode 進行遮罩形變響應
    if mode == "low":
        return h
    elif mode == "high":
        return 1 - h
    elif mode == "bandpass":
        # 帶通：只保留 d0 到 d1 之間的頻率 (利用布林矩陣轉成 0 與 1)
        return ((dist >= d0) & (dist <= d1)).astype(float)
    elif mode == "bandreject":
        # 帶拒：濾除 d0 到 d1 之間的頻率
        return (~((dist >= d0) & (dist <= d1))).astype(float)
    return h

def ensure_grayscale(img):
    """工具函式：確保影像為單通道灰階圖（AOI 許多特徵演算法必備前處理）"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

# ==========================================
# 2. Streamlit 介面與側邊欄設定
# ==========================================

st.set_page_config(layout="wide", page_title="AOI 影像處理演算法實驗室")

with st.sidebar:
    st.title("🎛️ 演算法控制面板")
    uploaded_file = st.file_uploader("上傳實驗影像", type=["jpg", "png", "tif"])
    
    st.divider()
    st.subheader("🧪 處理流程 (可自訂順序)")
    
    # 宣告 AOI 實驗室支援的演算法流水線清單
    step_options = [
        "轉為灰階", "顏色翻轉", "高斯模糊 (濾波)", "中值濾波 (去噪)", 
        "Canny 邊緣檢測", "二值化處理", "Hough 直線偵測", 
        "方向性邊緣偵測", "圓圈檢測", "低通濾波", "高通濾波", "帶通/帶拒"
    ]
    selected_steps = st.multiselect(
        "請依序選擇處理步驟：", step_options,
        help="演算法執行的先後順序會極大影響 AOI 檢測結果，例如通常先降噪再做邊緣偵測。"
    )
    
    st.divider()
    do_hist = st.checkbox("顯示像素直方圖分析及頻譜分析圖")

# ==========================================
# 3. 主畫面核心流水線邏輯
# ==========================================

st.title("🔬 自動化光學檢測 (AOI) 演算法平台")

if uploaded_file is not None:
    # 讀取上傳的影像檔案並解碼為 OpenCV 矩陣 (BGR 格式)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)
    img_current = img_original.copy() # 複製一份作為流水線的動態暫存容器

    # 依序執行使用者選擇的動態流水線步驟
    for step in selected_steps:
        
        if step == "轉為灰階":
            img_current = ensure_grayscale(img_current)

        elif step == "顏色翻轉":
            img_current = cv2.bitwise_not(img_current)
            
        elif step == "高斯模糊 (濾波)":
            k = st.sidebar.slider("高斯核大小", 1, 15, 5, step=2) # 核大小必須為奇數
            sig = st.sidebar.slider("SigmaX", 0.0, 5.0, 0.0)
            img_current = cv2.GaussianBlur(img_current, (k, k), sig)
        
        elif step == "中值濾波 (去噪)":
            k = st.sidebar.slider("中值濾波核大小", 3, 15, 5, step=2) # 針對椒鹽雜訊效果極佳
            img_current = cv2.medianBlur(img_current, k)
            
        elif step == "Canny 邊緣檢測":
            temp = ensure_grayscale(img_current) # 自動防錯，Canny 必須在灰階下運作
            t1 = st.sidebar.slider("Canny 低閾值", 0, 255, 50)
            st.sidebar.caption("低於此值會被排除")
            t2 = st.sidebar.slider("Canny 高閾值", 0, 255, 150)
            st.sidebar.caption("高於此值視為強邊緣")
            img_current = cv2.Canny(temp, t1, t2)
            
        elif step == "二值化處理":
            temp = ensure_grayscale(img_current)
            thresh = st.sidebar.slider("二值化門檻值 (Thresh)", 0, 255, 127)
            _, img_current = cv2.threshold(temp, thresh, 255, cv2.THRESH_BINARY)

        elif step == "Hough 直線偵測":
            # 霍夫變換直線偵測前，通常需要先取得二值邊緣圖
            edges = img_current if len(img_current.shape) == 2 else cv2.Canny(ensure_grayscale(img_current), 50, 150)
            
            h_thresh = st.sidebar.slider("Hough 投票閾值", 10, 200, 100)
            min_len = st.sidebar.slider("最小線段長度", 1, 200, 50)
            max_gap = st.sidebar.slider("最大線段間隙", 1, 50, 10)
            
            # 將結果圖轉為彩色的 BGR 畫布，以便繪製綠色的偵測直線
            line_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(edges.shape) == 2 else edges.copy()
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, h_thresh, minLineLength=min_len, maxLineGap=max_gap)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # 繪製綠色直線
            img_current = line_img

        elif step == "圓圈檢測":
            gray = ensure_grayscale(img_current)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2) # 霍夫圓對雜訊極敏感，內部強制平滑降噪
            
            p2 = st.sidebar.slider("圓心投票閾值 (param2)", 10, 100, 30)
            min_dist = st.sidebar.slider("圓心最小距離", 10, 200, 50)
            r_limit = st.sidebar.slider("半徑範圍(Min, Max)", 0, 500, (10, 100))
    
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, 
                param1=100, param2=p2, minRadius=r_limit[0], maxRadius=r_limit[1]
            )
    
            res_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(res_img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 畫外圓 (綠色)
                    cv2.circle(res_img, (i[0], i[1]), 2, (0, 0, 255), 3)    # 畫圓心 (紅色)
            img_current = res_img

        elif step in ["低通濾波", "高通濾波", "帶通/帶拒"]:
            img_gray = ensure_grayscale(img_current)
            rows, cols = img_gray.shape
    
            f_type = st.sidebar.radio("濾波器類型", ["ideal", "gaussian", "butterworth"])
            mode = "low" if step == "低通濾波" else "high" if step == "高通濾波" else st.sidebar.radio("模式", ["bandpass", "bandreject"])
    
            d0 = st.sidebar.slider("截止頻率 (D0)", 1, 300, 30)
            d1 = st.sidebar.slider("高階截止 (D1, 僅限帶通/帶拒)", d0, 300, 60)
            n = st.sidebar.slider("巴特沃斯階數 (n)", 1, 5, 2)
        
            # 核心頻域運算三大步：1. FFT 2. 乘上遮罩 3. 逆反傅立葉 IFFT
            fshift = get_fft_shift(img_gray)
            mask = create_mask(rows, cols, mode, f_type, d0, d1, n)
            img_current = inverse_fft(fshift * mask)

        elif step == "方向性邊緣偵測":
            temp = ensure_grayscale(img_current)
            direction = st.sidebar.selectbox("選擇偵測方向", ["水平 (0°)", "垂直 (90°)", "垂直 + 水平", "+45°", "-45°"])
            blend_weight = st.sidebar.slider("水平/垂直融合比例", 0.0, 1.0, 0.5)
            brightness_offset = st.sidebar.slider("邊緣亮度補償", 0, 100, 0)
    
            # 利用 Dictionary 映射各方向的卷積核 (Kernel Matrix)
            kernels = {
                "水平 (0°)": np.array([[-1, -1, -1], [ 2,  2,  2], [-1, -1, -1]]),
                "垂直 (90°)": np.array([[-1,  2, -1], [-1,  2, -1], [-1,  2, -1]]),
                "+45°":       np.array([[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]]),
                "-45°":       np.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]])
            }
    
            if direction == "垂直 + 水平":
                # 使用 CV_16S (16位有符號整數) 進行濾波，防止邊緣負值被截斷歸零
                res_h = cv2.filter2D(temp, cv2.CV_16S, kernels["水平 (0°)"])
                res_v = cv2.filter2D(temp, cv2.CV_16S, kernels["垂直 (90°)"])
                # 轉回 8位無符號整數 (取絕對值)
                abs_h = cv2.convertScaleAbs(res_h)
                abs_v = cv2.convertScaleAbs(res_v)
                # 依比例動態融合水平與垂直特徵
                img_current = cv2.addWeighted(abs_h, 1 - blend_weight, abs_v, blend_weight, brightness_offset)
            else:
                # 單一方向處理
                filtered = cv2.filter2D(temp, cv2.CV_16S, kernels[direction])
                img_current = cv2.convertScaleAbs(filtered + brightness_offset)

    # ==========================================
    # 4. 主畫面結果呈現與數據可視化
    # ==========================================
    
    # 畫面佈局：左邊 2/3 放影像對比，右邊 1/3 放統計圖表
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🖼️ 影像對比結果")
        if selected_steps:
            st.caption(f"🚀 當前 AOI 執行路徑: 原圖 ➔ {' ➔ '.join(selected_steps)}")
        
        sub_col_orig, sub_col_proc = st.columns(2)
        
        with sub_col_orig:
            # OpenCV 為 BGR 格式，Streamlit 顯示需轉換為 RGB
            st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB), caption="原始影像 (Original)", use_container_width=True)
            
        with sub_col_proc:
            # 判斷目前處理後的影像通道數以正確轉換格式呈現
            display_img = cv2.cvtColor(img_current, cv2.COLOR_BGR2RGB) if len(img_current.shape) == 3 else img_current
            st.image(display_img, caption="處理後影像 (Processed)", use_container_width=True)

    with col_right:
        if do_hist:
            # ----- 像素直方圖分析 -----
            st.subheader("📊 像素直方圖")
            fig, ax = plt.subplots()
            
            if len(img_current.shape) == 2:
                # 灰階圖：利用 .ravel() 將二維矩陣拉平為一維向量以便計算直方圖響應
                ax.hist(img_current.ravel(), 256, [0, 256], color='black')
                ax.set_title("Grayscale Histogram")
            else:
                # 彩色圖：依序撈取 B, G, R 通道計算色彩分佈
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([img_current], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col)
                ax.set_title("RGB Histogram")
            
            ax.set_xlim([0, 256])
            st.pyplot(fig)
            plt.close(fig) # 釋放內存避免警告
            
            # ----- 頻域二維傅立葉光譜分析 -----
            st.subheader("🌌 頻域分析 (FFT)")
            fft_input = ensure_grayscale(img_current) # 必須在灰階訊號下執行 FFT
            
            f = np.fft.fft2(fft_input)
            fshift = np.fft.fftshift(f)
            
            # 計算振幅譜 (Magnitude Spectrum)，+1 避免 log(0)，乘上 20 做分貝尺度轉換
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
            fig_fft, ax_fft = plt.subplots()
            ax_fft.imshow(magnitude_spectrum, cmap='gray')
            ax_fft.set_title('Frequency Spectrum')
            ax_fft.axis('off') # 隱藏座標軸軸線增加美觀度
            st.pyplot(fig_fft)
            plt.close(fig_fft)
        else:
            st.info("👈 勾選側邊欄「顯示像素直方圖分析及頻譜分析圖」可查看深度的影像數據分佈。")

else:
    st.warning("請先上傳圖片以開始 AOI 演算法測試。")
