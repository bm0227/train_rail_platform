import streamlit as st
import os
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 이미지를 1차원 벡터로 변환하는 함수
def image_to_vector(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # RGB 이미지를 흑백으로 변환
    img_resized = cv2.resize(img_gray, (128, 128))  # 이미지 크기 조정 (원하는 크기로 설정)
    img_vector = img_resized.flatten()  # 이미지를 1차원 벡터로 변환
    return img_vector

# One-Class SVM 모델 생성
def create_svm_model(train_path):
    # 학습할 정상 이미지 데이터를 로드
    normal_images = []
    for filename in os.listdir(train_path):
        img_path = os.path.join(train_path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 흑백 이미지로 변환
            img = cv2.resize(img, (128, 128))  # 이미지 크기 조정 (원하는 크기로 설정)
            img_vector = img.flatten()  # 이미지를 1차원 벡터로 변환
            normal_images.append(img_vector)
    normal_images = np.array(normal_images)
    
    # One-Class SVM 모델 생성
    svm_model = Pipeline([
        ("scaler", StandardScaler()),  # 특성 스케일링
        ("svm", OneClassSVM(kernel='rbf', gamma='auto')),  # One-Class SVM 모델
    ])
    
    # 모델 학습
    svm_model.fit(normal_images)
    
    return svm_model

# Streamlit 애플리케이션 정의
st.title('이미지 이상 탐지 웹 애플리케이션')

# 학습할 데이터 폴더를 선택할 수 있는 파일 업로더 추가
train_path = st.sidebar.text_input('학습할 데이터 폴더 경로 입력', '/path/to/your/training/images/')

# 학습할 데이터 폴더 경로가 입력되었는지 확인
if not os.path.isdir(train_path):
    st.error('입력한 경로가 유효하지 않습니다.')
    st.stop()

# One-Class SVM 모델 생성
svm_model = create_svm_model(train_path)

uploaded_file = st.file_uploader("이미지 파일 업로드", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 업로드한 이미지 보기
    image = np.array(cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1))
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
