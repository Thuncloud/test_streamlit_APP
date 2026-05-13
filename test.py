import streamlit as st
import cv2
import numpy as np
from streamlit_folium import st_folium
import folium
from PIL import Image

st.title("AOI影像處理")
st.write("選擇操作模式並上傳圖片")

option = st.sidebar.selectbox("選擇一個選項：", ["選項1", "選項2", "選項3"])
uploaded_file = st.sidebar.file_uploader("上傳影像進行處理...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    match option:
            case "選項1":
                st.write("你選擇1")
                # 文字輸入框
                name1 = st.text_input("輸入你的名字", value="你的名字")
                st.write(f"你好，{name1}!")

            case "選項2":
                st.write("你選擇2")
                st.image(image)
                
            case "選項3":
                st.write("你選擇3")
            

