import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 引入 pandas 用來精美呈現特徵數據表

# ==========================================
# 1. 核心演算法工具函式
# ==========================================

def get_fft_shift(img):
    """執行快速傅立葉轉換 (FFT) 並將低頻成分位移至中心"""
    f = np.fft.fft2(img)
    return np.fft.fftshift(f)

def inverse_fft(fshift):
    """執行反傅立葉轉換 (IFFT) 將頻域訊號轉回空間域影像"""
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.uint8(cv2.normalize(np.abs(img_back), None, 0, 255, cv2.NORM_MINMAX))

def create_mask(rows, cols, mode="low", filter_type="ideal", d0=30, d1=60, n=2):
    """建立頻域濾波遮罩 (Mask)"""
    crow, ccol = rows // 2, cols // 2
    u = np.linspace(0, rows - 1, rows)
    v = np.linspace(0, cols - 1, cols)
    V, U = np.meshgrid(v, u)
    dist = np.sqrt((U - crow)**2 + (V - ccol)**2)
    dist = np.where(dist == 0, 1e-5, dist)

    if filter_type == "ideal":
        h = np.zeros((rows, cols))
        h[dist <= d0] = 1
    elif filter_type == "gaussian":
        h = np.exp(-(dist**2) / (2 * (d0**2)))
    elif filter_type == "butterworth":
        h = 1 / (1 + (dist / d0)**(2 * n))

    if mode == "low":
        return h
    elif mode == "high":
        return 1 - h
    elif mode == "bandpass":
        return ((dist >= d0) & (dist <= d1)).astype(float)
    elif mode == "bandreject":
        return (~((dist >= d0) & (dist <= d1))).astype(float)
    return h

def ensure_grayscale(img):
    """工具函式：確保影像為單通道灰階圖"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img


# ==========================================
# 2. Streamlit 介面初始化與 Session State 建立
# ==========================================
st.set_page_config(layout="wide", page_title="AOI 影像處理演算法實驗室")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = []

# 用來存放最後一階段特徵提取的數據，方便在主畫面下方繪製表格
features_data = [] 


# ==========================================
# 3. 側邊欄：分門別類的動態流水線控制
# ==========================================
with st.sidebar:
    st.title("🎛️ 模組化演算法面板")
    uploaded_file = st.file_uploader("上傳實驗影像", type=["jpg", "png", "tif"])
    
    st.divider()
    st.subheader("🧪 演算法分類清單")
    st.caption("點擊各類別展開，並按 ➕ 依序加入你的檢測流水線：")
    
    # --- 分類 1：前處理與去噪 ---
    with st.expander("✨ 1. 影像前處理與降噪", expanded=False):
        opt_pre = st.selectbox("選擇前處理：", ["請選擇...", "轉為灰階", "顏色翻轉", "高斯模糊 (濾波)", "中值濾波 (去噪)"], key="sel_pre")
        if st.button("➕ 加入流水線", key="btn_pre") and opt_pre != "請選擇...":
            st.session_state.pipeline.append(opt_pre)
            
    # --- 分類 2：邊緣與特徵提取 ---
    with st.expander("📐 2. 空間域與特徵偵測", expanded=False):
        # 這裡加入了【AOI 特徵分析與過濾】選項！
        opt_feat = st.selectbox("選擇特徵偵測：", ["請選擇...", "Canny 邊緣檢測", "二值化處理", "Hough 直線偵測", "方向性邊緣偵測", "圓圈檢測", "AOI 特徵分析與過濾"], key="sel_feat")
        if st.button("➕ 加入流水線", key="btn_feat") and opt_feat != "請選擇...":
            st.session_state.pipeline.append(opt_feat)
            
    # --- 分類 3：頻域分析 ---
    with st.expander("🌌 3. 傅立葉頻域濾波", expanded=False):
        opt_freq = st.selectbox("選擇頻域濾波：", ["請選擇...", "低通濾波", "高通濾波", "帶通/帶拒"], key="sel_freq")
        if st.button("➕ 加入流水線", key="btn_freq") and opt_freq != "請選擇...":
            st.session_state.pipeline.append(opt_freq)

    # --- 流水線管理面板 ---
    st.divider()
    st.subheader("📋 目前排定的 AOI 流程")
    
    if st.session_state.pipeline:
        for idx, step in enumerate(st.session_state.pipeline):
            col_step_name, col_step_del = st.columns([4, 1])
            with col_step_name:
                st.markdown(f"**{idx + 1}.** {step}")
            with col_step_del:
                if st.button("❌", key=f"del_{idx}", help=f"移除第 {idx+1} 步：{step}"):
                    st.session_state.pipeline.pop(idx)
                    st.rerun()
        
        st.write("")
        if st.button("🔄 重設所有步驟", type="secondary", use_container_width=True):
            st.session_state.pipeline = []
            st.rerun()
    else:
        st.info("目前尚未選擇步驟，請從上方分類加入。")
        
    st.divider()
    do_hist = st.checkbox("顯示像素直方圖分析及頻譜分析圖")


# ==========================================
# 4. 主畫面核心流水線邏輯 (依排定步驟執行)
# ==========================================
st.title("🔬 自動化光學檢測 (AOI) 演算法平台")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, 1)
    img_current = img_original.copy()

    for step in st.session_state.pipeline:
        
        if step == "轉為灰階":
            img_current = ensure_grayscale(img_current)

        elif step == "顏色翻轉":
            img_current = cv2.bitwise_not(img_current)
            
        elif step == "高斯模糊 (濾波)":
            k = st.sidebar.slider("高斯核大小", 1, 15, 5, step=2)
            sig = st.sidebar.slider("SigmaX", 0.0, 5.0, 0.0)
            img_current = cv2.GaussianBlur(img_current, (k, k), sig)
        
        elif step == "中值濾波 (去噪)":
            k = st.sidebar.slider("中值濾波核大小", 3, 15, 5, step=2)
            img_current = cv2.medianBlur(img_current, k)
            
        elif step == "Canny 邊緣檢測":
            temp = ensure_grayscale(img_current)
            t1 = st.sidebar.slider("Canny 低閾值", 0, 255, 50)
            t2 = st.sidebar.slider("Canny 高閾值", 0, 255, 150)
            img_current = cv2.Canny(temp, t1, t2)
            
        elif step == "二值化處理":
            temp = ensure_grayscale(img_current)
            st.sidebar.markdown("---")
            bin_type = st.sidebar.radio("二值化模式", ["傳統固定門檻", "自適應門檻 (過濾亮度干擾)"])
            
            if bin_type == "傳統固定門檻":
                thresh = st.sidebar.slider("二值化門檻值", 0, 255, 127)
                _, img_current = cv2.threshold(temp, thresh, 255, cv2.THRESH_BINARY)
            elif bin_type == "自適應門檻 (過濾亮度干擾)":
                method_opt = st.sidebar.selectbox("自適應計算方法", ["區域平均加權 (MEAN)", "高斯分佈加權 (GAUSSIAN)"])
                adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if "MEAN" in method_opt else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                block_size = st.sidebar.slider("局部核大小 (Block Size)", 3, 99, 11, step=2)
                c_value = st.sidebar.slider("微調常數 (C)", -20, 20, 2)
                img_current = cv2.adaptiveThreshold(temp, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c_value)

        elif step == "形態學斷開運算 (Opening)":
            temp = ensure_grayscale(img_current)
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("🧮 結構元素設定 (Structuring Element)")
            
            # 1. 讓使用者「自行選擇形狀」
            se_shape_opt = st.sidebar.selectbox(
                "選擇結構元素形狀：", 
            ["矩形 (RECT)", "橢圓形/圓形 (ELLIPSE)", "十字形 (CROSS)"]
            )
        
            # 對應 OpenCV 的常數
            if "RECT" in se_shape_opt:
                se_shape = cv2.MORPH_RECT
              elif "ELLIPSE" in se_shape_opt:
                  se_shape = cv2.MORPH_ELLIPSE
              else:
                  se_shape = cv2.MORPH_CROSS
        
              # 2. 讓使用者「自行選擇大小」 (必須 >= 1)
              se_size = st.sidebar.slider("結構元素大小 (Size)", 1, 51, 5, step=2)
    
              # 3. 根據選擇建立結構元素
              kernel = cv2.getStructuringElement(se_shape, (se_size, se_size))
    
              # 執行形態學運算 (Opening: 先侵蝕後膨脹，用來斷開微小噪點)
              img_current = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
        
        elif step == "AOI 特徵分析與過濾":
            # 【全新實作第 2 點需求】：物件篩選與特徵量化
            # 確保輸入是二值化影像 (單通道)
            binary_src = ensure_grayscale(img_current)
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 特徵過濾條件設定")
            
            # 設定面積過濾閘門 (Filter Gate)
            min_area = st.sidebar.slider("最小面積門檻 (過濾噪點)", 0, 5000, 50)
            max_area = st.sidebar.slider("最大面積門檻", min_area, 100000, 50000)
            
            # 執行連通域分析 (Connected Components with Stats)
            # num_labels: 連通域數量, labels: 標記矩陣, stats: 包含[左, 上, 寬, 高, 面積], centroids: 中心點座標
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_src)
            
            # 建立一個乾淨的 BGR 畫布，用來繪製通過篩選的目標
            output_canvas = cv2.cvtColor(binary_src, cv2.COLOR_GRAY2BGR)
            features_data = [] # 清空前一次的紀錄
            
            object_id = 1
            # 索引 0 是背景 (Background)，AOI 物件要從 1 開始遍歷
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]   # 撈取該物件的面積
                
                # 【條件篩選控制門檻】
                if min_area <= area <= max_area:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    cx, cy = centroids[i]           # 質心座標 (中心點)
                    aspect_ratio = float(w) / h     # 計算長寬比
                    
                    # 1. 在畫布上標記黃色外框 (Bounding Box)
                    cv2.rectangle(output_canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    
                    # 2. 畫上紅色的圓心 (質心)
                    cv2.circle(output_canvas, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                    
                    # 3. 在圖面上寫上物件編號 ID
                    cv2.putText(output_canvas, f"ID:{object_id}", (x, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 4. 紀錄特徵數據，準備餵給 Dataframe 顯示
                    features_data.append({
                        "物件 ID": f"ID_{object_id}",
                        "面積 (pixels)": area,
                        "中心點 X (cx)": round(cx, 1),
                        "中心點 Y (cy)": round(cy, 1),
                        "外框寬度 (w)": w,
                        "外框高度 (h)": h,
                        "長寬比 (Aspect Ratio)": round(aspect_ratio, 2)
                    })
                    object_id += 1
                    
            img_current = output_canvas

        elif step == "Hough 直線偵測":
            edges = img_current if len(img_current.shape) == 2 else cv2.Canny(ensure_grayscale(img_current), 50, 150)
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
            gray = ensure_grayscale(img_current)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            p2 = st.sidebar.slider("圓心投票閾值 (param2)", 10, 100, 30)
            min_dist = st.sidebar.slider("圓心最小距離", 10, 200, 50)
            r_limit = st.sidebar.slider("半徑範圍", 0, 500, (10, 100))
    
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, 
                param1=100, param2=p2, minRadius=r_limit[0], maxRadius=r_limit[1]
            )
    
            res_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(res_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(res_img, (i[0], i[1]), 2, (0, 0, 255), 3)
            img_current = res_img

        elif step in ["低通濾波", "高通濾波", "帶通/帶拒"]:
            img_gray = ensure_grayscale(img_current)
            rows, cols = img_gray.shape
            f_type = st.sidebar.radio("濾波器類型", ["ideal", "gaussian", "butterworth"])
            mode = "low" if step == "低通濾波" else "high" if step == "高通濾波" else st.sidebar.radio("模式", ["bandpass", "bandreject"])
            d0 = st.sidebar.slider("截止頻率 (D0)", 1, 300, 30)
            d1 = st.sidebar.slider("高階截止 (D1)", d0, 300, 60)
            n = st.sidebar.slider("巴特沃斯階數 (n)", 1, 5, 2)
        
            fshift = get_fft_shift(img_gray)
            mask = create_mask(rows, cols, mode, f_type, d0, d1, n)
            img_current = inverse_fft(fshift * mask)

        elif step == "方向性邊緣偵測":
            temp = ensure_grayscale(img_current)
            direction = st.sidebar.selectbox("選擇偵測方向", ["水平 (0°)", "垂直 (90°)", "垂直 + 水平", "+45°", "-45°"])
            blend_weight = st.sidebar.slider("水平/垂直融合比例", 0.0, 1.0, 0.5)
            brightness_offset = st.sidebar.slider("邊緣亮度補償", 0, 100, 0)
    
            kernels = {
                "水平 (0°)": np.array([[-1, -1, -1], [ 2,  2,  2], [-1, -1, -1]]),
                "垂直 (90°)": np.array([[-1,  2, -1], [-1,  2, -1], [-1,  2, -1]]),
                "+45°":       np.array([[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]]),
                "-45°":       np.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]])
            }
    
            if direction == "垂直 + 水平":
                res_h = cv2.filter2D(temp, cv2.CV_16S, kernels["水平 (0°)"])
                res_v = cv2.filter2D(temp, cv2.CV_16S, kernels["垂直 (90°)"])
                img_current = cv2.addWeighted(cv2.convertScaleAbs(res_h), 1 - blend_weight, cv2.convertScaleAbs(res_v), blend_weight, brightness_offset)
            else:
                filtered = cv2.filter2D(temp, cv2.CV_16S, kernels[direction])
                img_current = cv2.convertScaleAbs(filtered + brightness_offset)

    # ==========================================
    # 5. 數據與結果視覺化呈現
    # ==========================================
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🖼️ 影像對比結果")
        if st.session_state.pipeline:
            st.caption(f"🚀 當前 AOI 執行路徑: 原圖 ➔ {' ➔ '.join(st.session_state.pipeline)}")
        
        sub_col_orig, sub_col_proc = st.columns(2)
        with sub_col_orig:
            st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB), caption="原始影像 (Original)", use_container_width=True)
            
        with sub_col_proc:
            display_img = cv2.cvtColor(img_current, cv2.COLOR_BGR2RGB) if len(img_current.shape) == 3 else img_current
            st.image(display_img, caption="處理後影像 (Processed)", use_container_width=True)

    with col_right:
        if do_hist:
            st.subheader("📊 像素直方圖")
            fig, ax = plt.subplots()
            if len(img_current.shape) == 2:
                ax.hist(img_current.ravel(), 256, [0, 256], color='black')
                ax.set_title("Grayscale Histogram")
            else:
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([img_current], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col)
                ax.set_title("RGB Histogram")
            ax.set_xlim([0, 256])
            st.pyplot(fig)
            plt.close(fig)
            
            st.subheader("🌌 頻域分析 (FFT)")
            fft_input = ensure_grayscale(img_current)
            f = np.fft.fft2(fft_input)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
            fig_fft, ax_fft = plt.subplots()
            ax_fft.imshow(magnitude_spectrum, cmap='gray')
            ax_fft.set_title('Frequency Spectrum')
            ax_fft.axis('off')
            st.pyplot(fig_fft)
            plt.close(fig_fft)
        else:
            st.info("👈 勾選側邊欄可查看處理後的數據分佈。")

    # --- 【資料表輸出區】 ---
    # 如果流水線有執行特徵萃取，且有撈到數據，就在底部展示結構化 DataFrame
    if "AOI 特徵分析與過濾" in st.session_state.pipeline:
        st.divider()
        st.subheader("📈 檢測目標定量特徵數據表 (Quantified AOI Features)")
        if features_data:
            df = pd.DataFrame(features_data)
            st.dataframe(df, use_container_width=True)
            st.success(f"🎯 成功識別出 {len(features_data)} 個符合篩選條件的 AOI 特徵目標！")
        else:
            st.warning("⚠️ 當前畫面中沒有符合面積過濾範圍的物件，請調整側邊欄的面積拉桿。")

else:
    st.warning("請先上傳圖片以開始 AOI 演算法測試。")
