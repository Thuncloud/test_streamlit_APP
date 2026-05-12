import streamlit as st
import cv2
import numpy as np

st.title("Hello, Streamlit!")
st.write("這是一個簡單的範例，歡迎來到 Streamlit 的世界！")

# 文字輸入框
name1 = st.text_input("請輸入你的名字：", value="你的名字")
st.write(f"你好，{name1}！")

number1 = st.number_input("請輸入數字：", key = "num_1")
number2 = st.number_input("請輸入數字：", key = "num_2")

result1 = number1 + number2
st.write(f"和:{result1}")

# 按鈕
if st.button("計算乘積"):
    result2 = number1 * number2
    st.write(f"乘積:{result2}")

option = st.sidebar.selectbox("選擇一個選項：", ["選項1", "選項2", "選項3"])
st.write(f"你選擇了：{option}")

uploaded_file = st.file_uploader("上傳影像進行處理...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 第一步：將上傳的檔案轉為 byte 陣列
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # 第二步：使用 OpenCV 解碼成影像格式 (BGR)
    cv_image = cv2.imdecode(file_bytes, 1)
    
    # 第三步：處理影像 (例如轉灰階)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # 第四步：顯示 (注意：st.image 預設是 RGB，所以 OpenCV 影像要先轉回 RGB)
    st.image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), caption="OpenCV 讀取的影像")
