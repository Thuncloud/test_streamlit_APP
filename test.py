import streamlit as st

st.title("Hello, Streamlit!")
st.write("這是一個簡單的範例，歡迎來到 Streamlit 的世界！")

# 文字輸入框
name1 = st.text_input("請輸入你的名字：", value="你的名字")
st.write(f"你好，{name1}！")

number1 = st.number_input("請輸入數字：", key = "num_1")
number2 = st.number_input("請輸入數字：", key = "num_2")

# 按鈕
if st.button("計算和"):
    result = number1 + number2
    st.write(f"{result}")
