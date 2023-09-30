import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow import keras


# 데이터를 불러옵니다.
data = pd.read_csv("data/final_test.csv", parse_dates=["Open_time"])
data = data.sort_values("Open_time")

# 사용할 feature들을 선택합니다.
features = ["Open", "High", "Low", "Close"]

# 데이터를 스케일링합니다.
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# 데이터를 시퀀스로 구성합니다.
seq_length = 192
X, y = [], []

for i in range(len(data) - seq_length):
    X.append(data[features].iloc[i : i + seq_length].values)
    y.append(data[features].iloc[i + seq_length].values)

X = np.array(X)
y = np.array(y)

# 데이터를 학습셋과 테스트셋으로 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# LSTM 모델을 정의합니다.
model = keras.models.Sequential(
    [
        keras.layers.LSTM(
            50,
            activation="tanh",
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        ),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50, activation="tanh"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(y_train.shape[1]),
    ]
)

# 모델을 컴파일합니다.
model.compile(optimizer="adam", loss="mean_squared_error")

# 모델을 학습시킵니다.
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# 학습과 검증 손실을 그래프로 표현합니다.
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend(loc="upper right")
plt.show()

# 모델을 평가합니다.
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
