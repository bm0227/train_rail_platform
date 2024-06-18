import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os

# 이미지를 1차원 벡터로 변환하는 함수
def image_to_vector(image):
    img_gray = image.convert('L')  # 흑백으로 변환
    img_resized = img_gray.resize((128, 128))  # 이미지 크기 조정 (원하는 크기로 설정)
    img_array = np.array(img_resized)  # 이미지를 배열로 변환
    img_vector = img_array.flatten()  # 이미지를 1차원 벡터로 변환
    return img_vector

# Streamlit 애플리케이션 정의
def main():
    st.title('이미지 이상 탐지 웹 애플리케이션')
    
    # 저장된 모델 파일 경로
    model_filename = 'one_class_svm_model.pkl'
    
    # 모델 로드하기
    svm_model = joblib.load(model_filename)
    
    uploaded_file = st.file_uploader("이미지 파일 업로드", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # 업로드한 이미지 보기
        image = Image.open(uploaded_file)  # 이미지 열기
        st.image(image, caption='업로드된 이미지', use_column_width=True)
        
        # 이미지를 1차원 벡터로 변환
        img_vector = image_to_vector(image)
        img_vector = img_vector.reshape(1, -1)  # 모델에 입력할 수 있는 형태로 reshape
        
        # 이상 여부 예측
        prediction = svm_model.predict(img_vector)
        
        # 결과 출력
        if prediction == 1:
            st.write("이 이미지는 정상 이미지입니다.")
        else:
            st.write("이 이미지는 비정상 이미지입니다.")

if __name__ == '__main__':
    main()
