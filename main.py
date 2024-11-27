import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. 엑셀 데이터 로드
file_path = 'lotto.csv'  # 엑셀 파일 경로를 설정

# 데이터 로드
lotto_data = pd.read_csv(file_path)
print(f"Number of rows in lotto_data: {lotto_data.shape[0]}")

# 2. 결측값 확인 후 처리
# 결측값이 있는 행을 확인
print(lotto_data[lotto_data.isnull().any(axis=1)])

# 결측값을 0으로 채우기 (혹은 다른 방법으로 처리)
lotto_data = lotto_data.dropna()
print(f"Number of rows after filling NA: {lotto_data.shape[0]}")

# 데이터가 비어있는지 확인
if lotto_data.shape[0] == 0:
    print("Data is empty after preprocessing. Please check the data source.")
else:
    # 3. 데이터 전처리
    # 입력 데이터 (6개 번호)
    X = lotto_data.iloc[:, :6].values
    print(X)
    # 출력 데이터 (당첨 번호: 모든 번호를 대상으로 확률 계산)
    y = np.concatenate((lotto_data.iloc[:, :6].values, lotto_data.iloc[:, 6:].values), axis=1)

    # 4. 원-핫 인코딩 함수 정의
    def one_hot_encode(data, max_num=45):
        encoded = np.zeros((data.shape[0], max_num))
        for i, row in enumerate(data):
            for num in row:
                num = int(num)  # Ensure num is treated as an integer
                encoded[i, num - 1] = 1
        return encoded

    # 입력 데이터 원-핫 인코딩
    X_encoded = one_hot_encode(X)

    # 출력 데이터 원-핫 인코딩 (당첨 번호)
    y_encoded = one_hot_encode(y)

    # 5. 학습/테스트 데이터 분리
    print(f"X shape: {X_encoded.shape}")
    print(f"y shape: {y_encoded.shape}")

    if X_encoded.shape[0] == 0 or y_encoded.shape[0] == 0:
        print("Data is empty after encoding. Please check the encoding logic.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

        # 6. 딥러닝 모델 정의
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=45, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(45, activation='softmax')  # 1~45 번호에 대한 확률
        ])

        # 7. 모델 컴파일
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 8. 모델 학습
        history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

        # 9. 학습 결과 확인 및 예측
        predictions = model.predict(X_test)

        # 특정 테스트 데이터에 대해 높은 확률을 가지는 번호 확인
        predicted_numbers = np.argsort(predictions[0])[-6:] + 1  # 가장 높은 확률의 6개 번호
        print("추천 번호:", predicted_numbers)

        # 10. 모델 저장
        model.save('lotto_model.h5')