import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 核心演算法 =================

def apply_frequency_filter(img, filter_type, mode, d0, d1=60, n=2):
    """通用頻域濾波器"""
    # 頻域運算必須是單通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 建立距離矩陣
    u = np.linspace(0, rows - 1, rows)
    v = np.linspace(0, cols - 1, cols)
    V, U = np.meshgrid(v, u)
    dist = np.sqrt((U - crow)**2 + (V - ccol)**2)
    dist = np.where(dist == 0, 1e-5, dist)

    # 遮罩 H(u,v)
    if filter_type == "理想":
        h = np.zeros((rows, cols))
        h[dist <= d0] = 1
    elif filter_type == "高斯":
        h = np.exp(-(dist**2) / (2 * (d0**2)))
    elif filter_type == "巴特沃斯":
        h = 1 / (1 + (dist / d0)**(2 * n))
    else:
        h = np.ones((rows, cols))

    # 模式變換
    if mode == "低通": mask = h
    elif mode == "高通": mask = 1 - h
    elif mode == "帶通":
        mask = np.zeros((rows, cols))
        mask[(dist >= d0) & (dist <= d1)] = 1
    elif mode == "帶拒":
        mask = np.ones((rows, cols))
        mask[(dist >= d0) & (dist <= d1)] = 0
    else: mask = h

    # 反轉換
    f_res = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(f_res))
    img_real = np.abs(img_back)
    return np.uint8(cv2.normalize(img_real, None, 0, 255, cv2.NORM_MINMAX))

# ================= 2. 介面設定 =================

st.set_page_config(layout="wide", page_title="AOI 演算法平台")
st.title("🔬 AOI 影像演算法綜合平台")

with st.sidebar:
    st.header("控制面板")
    uploaded_file = st.file_uploader("上傳影像", type=['png', 'jpg', 'tif'])
    
    # 你的功能清單
    all_ops = [
        "轉為灰階", "二值化", "顏色翻轉", 
        "低通濾波", "高通濾波", "帶通濾波", "帶拒濾波",
        "Canny邊緣偵測", "霍夫直線", "霍夫圓形"
    ]
    selected_ops = st.multiselect("處理流程", all_ops)

    st.divider()
    f_curve = st.selectbox("濾波器曲線", ["理想", "高斯", "巴特沃斯"])
    d0_val = st.slider("D0 (截止頻率)", 1, 200, 30)
    d1_val = st.slider("D1 (帶通/帶拒)", d0_val, 300, 60)
    b_order = st.slider("巴特沃斯階數 n", 1, 5, 2)
    show_plots = st.checkbox("顯示分析圖表", value=True)

# ================= 3. 處理邏輯 =================

if uploaded_file:
    # 讀取
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_origin = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_proc = img_origin.copy()

    for op in selected_ops:
        try:
            if op == "轉為灰階":
                if len(img_proc.shape) == 3:
                    img_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
            
            elif op == "二值化":
                g = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY) if len(img_proc.shape) == 3 else img_proc
                _, img_proc = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
            
            elif op == "顏色翻轉":
                img_proc = cv2.bitwise_not(img_proc)

            elif "濾波" in op:
                m_name = op.replace("濾波", "")
                img_proc = apply_frequency_filter(img_proc, f_curve, m_name, d0_val, d1_val, b_order)

            elif op == "Canny邊緣偵測":
                img_proc = cv2.Canny(img_proc, 100, 200)

            elif op == "霍夫直線":
                ed = cv2.Canny(img_proc, 50, 150)
                lines = cv2.HoughLinesP(ed, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
                canvas = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2BGR) if len(img_proc.shape) == 2 else img_proc.copy()
                if lines is not None:
                    for l in lines:
                        x1, y1, x2, y2 = l[0]
                        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img_proc = canvas

            elif op == "霍夫圓形":
                g_c = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY) if len(img_proc.shape) == 3 else img_proc
                circles = cv2.HoughCircles(g_c, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
                canvas = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2BGR) if len(img_proc.shape) == 2 else img_proc.copy()
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        cv2.circle(canvas, (i[0], i[1]), i[2], (0, 255, 0), 2)
                img_proc = canvas
        except Exception as e:
            st.error(f"步驟 {op} 出錯: {e}")

    # 顯示
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("原始影像")
        st.image(img_origin, channels="BGR", width=500)
    with c2:
        st.subheader("處理後影像")
        st.image(img_proc, width=500)

    # 圖表
    if show_plots:
        st.divider()
        h_col, f_col = st.columns(2)
        with h_col:
            st.write("📊 直方圖")
            fig1, ax1 = plt.subplots(figsize=(5,3))
            if len(img_proc.shape) == 2:
                ax1.hist(img_proc.ravel(), bins=256, range=(0, 256), color='gray')
            else:
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([img_proc], [i], None, [256], [0, 256])
                    ax1.plot(hist, color=col)
            st.pyplot(fig1)
        with f_col:
            st.write("🌌 頻譜圖")
            g_f = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY) if len(img_proc.shape) == 3 else img_proc
            f_s = np.fft.fftshift(np.fft.fft2(g_f))
            mag = 20 * np.log(np.abs(f_s) + 1)
            fig2, ax2 = plt.subplots(figsize=(5,3))
            ax2.imshow(mag, cmap='gray')
            ax2.axis('off')
            st.pyplot(fig2)
else:
    st.info("請上傳影像。")
