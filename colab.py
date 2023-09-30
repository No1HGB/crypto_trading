# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import h5py
from tensorflow import keras

# from google.colab import files


class DiskDataGenerator(keras.utils.Sequence):
    def __init__(self, filename, batch_size=32, is_train=True):
        self.filename = filename
        self.batch_size = batch_size

        with h5py.File(self.filename, "r") as f:
            self.length = len(f["X_train" if is_train else "X_test"])

    def __len__(self):
        return int(np.ceil(self.length / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size

        with h5py.File(self.filename, "r") as f:
            X_batch = f["X_train"][start:end]
            y_batch = f["y_train"][start:end]

        return X_batch, y_batch


# 데이터 불러오기
data = pd.read_csv(
    "drive/My Drive/Colab Notebooks/final.csv",
    parse_dates=["Open_time"],
)
data = data.sort_values("Open_time")

# 사용할 feature들을 선택합니다.
features = ["Open", "High", "Low", "Close"]

# 데이터를 스케일링합니다.
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# 데이터를 시퀀스로 저장합니다.
seq_length = 192
filename = "sequence_data.h5"

with h5py.File(filename, "w") as f:
    X_train_dset = f.create_dataset(
        "X_train",
        (0, seq_length, len(features)),
        maxshape=(None, seq_length, len(features)),
        compression="gzip",
    )
    y_train_dset = f.create_dataset(
        "y_train",
        (0, len(features)),
        maxshape=(None, len(features)),
        compression="gzip",
    )
    X_test_dset = f.create_dataset(
        "X_test",
        (0, seq_length, len(features)),
        maxshape=(None, seq_length, len(features)),
        compression="gzip",
    )
    y_test_dset = f.create_dataset(
        "y_test", (0, len(features)), maxshape=(None, len(features)), compression="gzip"
    )

    for i in range(len(data) - seq_length):
        X_seq = data[features].iloc[i : i + seq_length].values[np.newaxis, :, :]
        y_seq = data[features].iloc[i + seq_length].values[np.newaxis, :]

        if i < int((len(data) - seq_length) * 0.8):  # 80% training
            X_train_dset.resize(X_train_dset.shape[0] + 1, axis=0)
            X_train_dset[-1:] = X_seq

            y_train_dset.resize(y_train_dset.shape[0] + 1, axis=0)
            y_train_dset[-1:] = y_seq
        else:  # 20% testing
            X_test_dset.resize(X_test_dset.shape[0] + 1, axis=0)
            X_test_dset[-1:] = X_seq

            y_test_dset.resize(y_test_dset.shape[0] + 1, axis=0)
            y_test_dset[-1:] = y_seq

# 모델 정의 및 컴파일
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
model.compile(optimizer="adam", loss="mean_squared_error")

# 데이터 분할 및 학습
train_generator = DiskDataGenerator(filename, batch_size=32, is_train=True)
val_generator = DiskDataGenerator(filename, batch_size=32, is_train=False)
history = model.fit(train_generator, epochs=100, validation_data=val_generator)

# 학습과 검증 손실을 그래프로 표현합니다.
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend(loc="upper right")

# 그래프를 파일로 저장합니다.
plt.savefig("loss_plot.png")

# 그래프를 화면에 표시 및 저장
plt.show()
# files.download('loss_plot_1.png')

# 모델 평가
with h5py.File(filename, "r") as f:
    y_test = f["y_test"][:]
    y_pred = model.predict(val_generator)

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# 모델 저장 및 다운로드
model.save("lstm_1.h5")
# files.download("lstm_1.h5")
