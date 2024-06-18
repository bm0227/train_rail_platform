import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os

# Streamlit 애플리케이션 정의
def main():
    st.title('AI-based system for railway condition recognition')
    
    uploaded_file = st.file_uploader("Upload image file", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # 업로드한 이미지 보기
        image = Image.open(uploaded_file)  # 이미지 열기
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("This is a Normal track.")

if __name__ == '__main__':
    main()
