from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import numpy as np
import os

# 훈련 이미지 데이터 path
train_path = '/cotent/train/'

# 이미지를 1차원 벡터로 변환하는 함수
def image_to_vector(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 흑백 이미지로 변환
    img = cv2.resize(img, (128, 128))  # 이미지 크기 조정 (원하는 크기로 설정)
    img_vector = img.flatten()  # 이미지를 1차원 벡터로 변환
    return img_vector

# 학습할 데이터 폴더에서 정상 이미지 데이터를 로드
normal_images = []
for filename in os.listdir(train_path):
    img_path = os.path.join(train_path, filename)
    if os.path.isfile(img_path):
        img_vector = image_to_vector(img_path)
        normal_images.append(img_vector)
normal_images = np.array(normal_images)

# One-Class SVM 모델 생성
svm_model = Pipeline([
    ("scaler", StandardScaler()),  # 특성 스케일링
    ("svm", OneClassSVM(kernel='rbf', gamma='auto')),  # One-Class SVM 모델
])

# 모델 학습
svm_model.fit(normal_images)

# 모델 저장하기
model_filename = 'one_class_svm_model.pkl'
joblib.dump(svm_model, model_filename)
print(f"모델이 {model_filename} 파일로 저장되었습니다.")
