from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
import numpy as np
import pickle
from get_cal_data import get_cal_data

data_dim = 2
timesteps = 8
num_classes = 256

[x_train, y_train_bool, x_val, y_val_bool] = get_cal_data()

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(512, return_sequences=True,
                input_shape=(timesteps, data_dim)))  # 返回维度为 32 的向量序列
# model.add(LSTM(1024, return_sequences=True))  # 返回维度为 32 的向量序列
model.add(LSTM(512))  # 返回维度为 32 的单个向量
model.add(Dense(num_classes, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train_bool,
          batch_size=1024, epochs=20,
          validation_data=(x_val, y_val_bool))
