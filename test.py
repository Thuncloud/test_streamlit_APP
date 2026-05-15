import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 核心演算法定義 (頻域與空域) =================

def get_fft_shift(img):
    """執行傅立葉轉換並移至中心"""
    f = np.fft.fft2(img)
    return np.fft.fftshift(f)

def inverse_fft(fshift):
    """執行反傅立葉轉換並正規化輸出"""
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_real = np.abs(img_back)
    return np.uint8(cv2.normalize(img_real, None, 0, 255, cv2.NORM_MINMAX))

def create_filter_mask(rows, cols, filter_type, mode, d0, d1=60, n=2):
    """建立各種頻域濾波器遮罩"""
    crow, ccol = rows // 2, cols // 2
    u = np.linspace(0, rows - 1, rows)
    v = np.linspace(0, cols - 1, cols)
    V, U = np.meshgrid(v, u)
    dist = np.sqrt((U - crow)**2 + (V - ccol)**2)
    dist = np.where(dist == 0, 1e-5, dist) # 避免除以零

    # 基本低通 H(u,v)
    if filter_type == "理想":
        h = np.zeros((rows, cols))
        h[dist <= d0] = 1
    elif filter_type == "高斯":
        h = np.exp(-(dist**2) / (2 * (d0**2)))
    elif filter_type == "巴特沃斯":
        h = 1 / (1 + (dist / d0)**(2 * n))

    # 模式變換
    if mode == "低通":
        return h
    elif mode == "高通":
        return 1 - h
    elif mode == "帶通":
        mask = np.zeros((rows, cols))
        mask[(dist >= d0) & (dist <= d1)] = 1
        return mask
    elif mode == "帶拒":
        mask = np.ones((rows, cols))
        mask[(dist >= d0) & (dist <= d1)] = 0
        return mask
    return h

# ================= 2. Streamlit 介面佈局 =================

st.set_page_config(layout="wide")
st.title("🔬 AOI 影像演算法綜合平台")

# 側邊欄：功能選擇與參數
with st.sidebar:
    st.header("演算法控制面板")
    uploaded_file = st.file_uploader("上傳實驗影像", type=['png', 'jpg', 'tif'])
    
    # 根據你的清單整合功能
    all_steps = [
        "原影像", "轉為灰階", "二值化", "顏色翻轉",
        "低通濾波", "高通濾波", "帶通濾波", "帶拒濾波",
        "Canny邊緣偵測", "霍夫直線", "霍夫圓形"
    ]
    selected_steps = st.multiselect("處理流程 (可自訂順序)", all_steps, default=["原影像"])

    # 動態顯示對應參數
    st.divider()
    filter_type = st.selectbox("濾波器曲線", ["理想", "高斯", "巴特沃斯"])
    d0 = st.sidebar.slider("截止頻率 (D0)", 1, 200, 30)
    d1 = st.sidebar.slider("高階截止 (D1, 帶通/帶拒用)", d0, 300, 60)
    b_n = st.sidebar.slider("巴特沃斯階數 (n)", 1, 5, 2)
    
    show_analysis = st.checkbox("顯示直方圖及頻譜分析圖", value=True)

# ================= 3. 主處理邏輯 =================

if uploaded_file:
    # 讀取原始影像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_origin = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_current = img_origin.copy()

    # 執行選定的處理步驟
    for step in selected_steps:
        try:
            if step == "轉為灰階":
                if len(img_current.shape) == 3:
                    img_current = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
            
            elif step == "二值化":
                temp_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
                _, img_current = cv2.threshold(temp_gray, 127, 255, cv2.THRESH_BINARY)
            
            elif step == "顏色翻轉":
                img_current = cv2.bitwise_not(img_current)

            elif step in ["低通濾波", "高通濾波", "帶通濾波", "帶拒濾波"]:
                # 強制轉灰階處理
                if len(img_current.shape) == 3:
                    img_current = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
                
                mode_name = step.replace("濾波", "")
                fshift = get_fft_shift(img_current)
                mask = create_filter_mask(img_current.shape[0], img_current.shape[1], filter_type, mode_name, d0, d1, b_n)
                img_current = inverse_fft(fshift * mask)

            elif step == "Canny邊緣偵測":
                img_current = cv2.Canny(img_current, 100, 200)

            elif step == "霍夫直線":
                # 霍夫變換通常需要 Canny 邊緣
                edges = cv2.Canny(img_current, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
                res = cv2.cvtColor(img_current, cv2.COLOR_GRAY2BGR) if len(img_current.shape)==2 else img_current.copy()
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img_current = res

            elif step == "霍夫圓形":
                temp_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
                blurred = cv2.medianBlur(temp_gray, 5)
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
                res = cv2.cvtColor(img_current, cv2.COLOR_GRAY2BGR) if len(img_current.shape)==2 else img_current.copy()
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        cv2.circle(res, (i[0], i[1]), i[2], (0, 255, 0), 2)
                img_current = res

        except Exception as e:
            st.error(f"步驟 '{step}' 執行出錯: {e}")

    # ================= 4. 結果顯示區 =================

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ 原影像")
        st.image(img_origin, channels="BGR")
    with col2:
        st.subheader("✨ 處理後影像")
        # 判斷是否為黑白或彩色顯示
        st.image(img_current)

    if show_analysis:
        st.divider()
        col_hist, col_fft = st.columns(2)
        
        with col_hist:
            st.subheader("📊 像素直方圖")
            fig_h, ax_h = plt.subplots()
            # 處理多通道或單通道直方圖
            if len(img_current.shape) == 2:
                ax_h.hist(img_current.ravel(), 256, [0, 256], color='gray')
            else:
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([img_current], [i], None, [256], [0, 256])
                    ax_h.plot(hist, color=col)
            st.pyplot(fig_h)

        with col_fft:
            st.subheader("🌌 頻域分析 (FFT)")
            temp_fft = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) if len(img_current.shape) == 3 else img_current
            f_shift = get_fft_shift(temp_fft)
            mag_spec = 20 * np.log(np.abs(f_shift) + 1)
            fig_f, ax_f = plt.subplots()
            ax_f.imshow(mag_spec, cmap='gray')
            ax_f.axis('off')
            st.pyplot(fig_f)

else:
    st.info("請上傳影像以開始處理。")
