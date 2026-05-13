import streamlit as st
import cv2
import numpy as np
from streamlit_folium import st_folium
import folium
from PIL import Image

def display_map_with_bounds(uploaded_file):
    img = Image.open(uploaded_file)
    width, height = img.size
    img_array = np.array(img)

    # 在建立 Map 時就設定好邊界
    m = folium.Map(
        crs='Simple', 
        location=[height/2, width/2], 
        zoom_start=1,
        min_zoom=0,
        # 直接在這裡定義最大邊界範圍
        max_bounds=True, 
        max_lat=height,
        min_lat=0,
        max_lon=width,
        min_lon=0,
        tiles=None
    )

    folium.raster_layers.ImageOverlay(
        image=img_array,
        bounds=[[0, 0], [height, width]],
        opacity=1.0
    ).add_to(m)

    # 移除會報錯的 m.set_max_bounds(...)
    # 使用 fit_bounds 確保初始畫面完美貼合
    m.fit_bounds([[0, 0], [height, width]])
    
    st_folium(m, width="100%", height=600)


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
            if uploaded_file is not None:
                    display_map_with_bounds(uploaded_file)
            
        case "選擇3":
            st.write("你選擇3")
            

