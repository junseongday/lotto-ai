import numpy as np
import tensorflow as tf

# 1. 모델 로드
model = tf.keras.models.load_model('improved_lotto_model.h5')

# 2. 입력 데이터 준비
# 예: 45개의 번호 중 일부를 원-핫 인코딩 형태로 준비
input_data = np.zeros((5, 45))  # 5개의 조합 입력 (5 x 45 크기)

# 입력 데이터를 랜덤하게 선택 (예: 첫 5개의 번호를 고정)
for i in range(5):  
    selected_numbers = np.random.choice(range(45), size=6, replace=False)  # 랜덤하게 6개 선택
    for num in selected_numbers:
        input_data[i, num] = 1

# 3. 예측 수행
predictions = model.predict(input_data)

# 4. 상위 5개의 조합 생성
top_5_combinations = []
for i in range(predictions.shape[0]):  # 각 조합에 대해 처리
    # 상위 6개의 번호 추출
    top_6_numbers = np.argsort(predictions[i])[-6:] + 1  # 1-based index로 변환
    top_6_numbers = np.sort(top_6_numbers)  # 번호 정렬
    top_5_combinations.append(top_6_numbers)

# 5. 결과 출력
print("추천 조합 5개:")
for idx, combination in enumerate(top_5_combinations):
    print(f"조합 {idx + 1}: {combination}")
