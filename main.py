import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 엑셀 데이터 로드
file_path = 'lotto.csv'  # 엑셀 파일 경로를 설정
lotto_data = pd.read_csv(file_path)

# 데이터 로드 후 행 수 확인
print(f"Number of rows in lotto_data: {lotto_data.shape[0]}")

# 2. 결측값 확인 및 처리
lotto_data = lotto_data.dropna()
if lotto_data.shape[0] == 0:
    raise ValueError("Data is empty after preprocessing. Please check the data source.")

# 3. 데이터 전처리
# 입력 데이터: 6개 번호
X = lotto_data.iloc[:, :6].values
# 출력 데이터: 모든 당첨 번호
y = np.concatenate((lotto_data.iloc[:, :6].values, lotto_data.iloc[:, 6:].values), axis=1)

# 특성 추가 (예: 번호 합계, 평균, 홀/짝 비율)
X_features = np.hstack((
    X,
    np.sum(X, axis=1).reshape(-1, 1),   # 번호 합계
    np.mean(X, axis=1).reshape(-1, 1),  # 번호 평균
    (np.sum(X % 2 == 1, axis=1) / 6).reshape(-1, 1)  # 홀수 비율
))

# 데이터 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_features)

# 원-핫 인코딩
def one_hot_encode(data, max_num=45):
    encoded = np.zeros((data.shape[0], max_num))
    for i, row in enumerate(data):
        for num in row:
            encoded[i, int(num) - 1] = 1
    return encoded

X_encoded = one_hot_encode(X)
y_encoded = one_hot_encode(y)

# 4. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# 5. 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_dim=X_encoded.shape[1], activation='relu'),
    tf.keras.layers.BatchNormalization(),  # 배치 정규화 추가
    tf.keras.layers.Dropout(0.1),          # 드롭아웃 추가 (30% 노드 비활성화)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(y_encoded.shape[1], activation='softmax')
])

# 6. 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=1000,  # 학습 반복 횟수 증가
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)  # 과적합 방지
    ]
)

# 8. 학습 결과 확인 및 예측
predictions = model.predict(X_test)
predicted_numbers = [np.argsort(pred)[-6:] + 1 for pred in predictions[:5]]  # 상위 6개 번호
print("추천 번호:", predicted_numbers)

# 9. 모델 저장
model.save('improved_lotto_model.h5')

# 10. 학습 시각화
# import matplotlib.pyplot as plt

# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.title("Model Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.show()
