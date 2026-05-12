import streamlit as st

st.title("Hello, Streamlit!")
st.write("這是一個簡單的範例，歡迎來到 Streamlit 的世界！")

# 文字輸入框
name1 = st.text_input("請輸入你的名字：", value="你的名字")
st.write(f"你好，{name1}！")

number1 = st.number_input("請輸入數字：", key = "num_1")
number2 = st.number_input("請輸入數字：", key = "num_2")

if number1 is not None:
    st.write(f"你輸入了 {number1}")
else:
    st.warning("目前尚未輸入數字喔！")

# 按鈕
if st.button("點擊我"):
    st.write("你剛剛點擊了按鈕！")
