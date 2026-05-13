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

match option:
        case "選項1":
            st.write("你選擇1")
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
            
        case "選擇2":
            st.write("你選擇2")
            
        case "選擇3":
            st.write("你選擇3")
            

def display_map_with_bounds(uploaded_file):
    # 讀取影像獲取寬高
    img = Image.open(uploaded_file)
    width, height = img.size
    img_array = np.array(img)

    # 1. 建立地圖物件，關鍵在於 min_zoom 的設定
    # 這裡的 min_zoom 通常設為 0 或 1，視顯示容器大小而定
    m = folium.Map(
        crs='Simple', 
        location=[height/2, width/2], 
        zoom_start=1,
        min_zoom=0,           # 限制最小縮放比例，防止縮太小看到邊界外
        max_bounds=True,      # 啟動邊界限制
        tiles=None,
        control_scale=True
    )

    # 2. 疊加圖片
    folium.raster_layers.ImageOverlay(
        image=img_array,
        bounds=[[0, 0], [height, width]],
        opacity=1.0
    ).add_to(m)

    # 3. 限制地圖的最大平移範圍，使用者無法拖出圖片區域
    m.set_Max_Bounds([[0, 0], [height, width]])

    # 4. 讓地圖初始狀態就完美貼合邊界
    m.fit_bounds([[0, 0], [height, width]])

    # 顯示地圖
    st_folium(m, width="100%", height=600)

if uploaded_file:
    display_map_with_bounds(uploaded_file)
