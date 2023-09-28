# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow import keras
import gc

# 데이터 불러오기
chunk_size = 64
data_iter = pd.read_csv(
    "drive/My Drive/Colab Notebooks/final.csv",
    parse_dates=["Open_time"],
    chunksize=chunk_size,
)

# 데이터 스케일링
seq_length = 32
scaler = MinMaxScaler()

# 사용할 feature 선택
features = ["Open", "High", "Low", "Close"]

# LSTM 모델 정의
model = keras.models.Sequential(
    [
        keras.layers.LSTM(
            50,
            activation="tanh",
            return_sequences=True,
            input_shape=(seq_length, len(features)),
        ),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50, activation="tanh"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(features)),
    ]
)

# 모델 컴파일
model.compile(optimizer="adam", loss="mean_squared_error")

# EarlyStopping 콜백 정의
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

for chunk in data_iter:
    chunk[features] = chunk[features].astype(np.float32)
    X, y = [], []

    for i in range(len(chunk) - seq_length):
        X.append(chunk[features].iloc[i : i + seq_length].values)
        y.append(chunk[features].iloc[i + seq_length].values)

    X = np.array(X)
    y = np.array(y)

    if len(X) > 0 and len(y) > 0:
        # 모델 학습
        model.fit(
            X,
            y,
            epochs=1,  # 각 청크마다 1 epoch만큼 학습
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stopping],
        )
    del X, y  # 리스트를 삭제하여 메모리를 해제합니다.
    gc.collect()  # 가비지 컬렉션을 수행합니다.

# 테스트셋 로딩
data_test = pd.read_csv(
    "drive/My Drive/Colab Notebooks/final.csv",
    parse_dates=["Open_time"],
)

# 테스트셋을 시퀀스로 변환 및 스케일링
X_test, y_test = [], []
for i in range(len(data_test) - seq_length):
    X_test.append(data_test[features].iloc[i : i + seq_length].values)
    y_test.append(data_test[features].iloc[i + seq_length].values)

X_test = np.array(X_test)
y_test = np.array(y_test)

# 모델 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")

# from google.colab import files
# files.download("lstm_1.h5")

# from google.colab import files
# files.download("lstm_1.h5")
