import tensorflow as tf
import numpy as np

# 모델 로드
model = tf.keras.models.load_model('lotto_model.h5')

# 원-핫 인코딩 함수
def one_hot_encode(data, max_num=45):
    encoded = np.zeros((1, max_num))  # 1개의 데이터
    for num in data:
        encoded[0, num - 1] = 1  # 해당 번호에 1을 할당
    return encoded

# 사용자가 입력한 로또 번호 (예: [1, 3, 5, 7, 10, 13])
user_input = [1, 3, 5, 7, 10, 13]

# 입력 데이터 원-핫 인코딩
X_user_encoded = one_hot_encode(user_input)

# 예측 수행
predictions = model.predict(X_user_encoded)

# 예측 결과에서 확률이 가장 높은 6개 번호를 추출
predicted_numbers = np.argsort(predictions[0])[-6:] + 1  # 가장 높은 확률의 6개 번호
print("추천 번호:", predicted_numbers)
