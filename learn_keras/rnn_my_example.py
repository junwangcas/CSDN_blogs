from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import SimpleRNN, Activation
from keras.optimizers import SGD
import numpy as np
import pickle
from get_cal_data import get_cal_data
from keras.optimizers import Adam

data_dim = 2
timesteps = 8
num_classes = 256

[x_train, y_train_bool, x_val, y_val_bool] = get_cal_data()

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    input_shape=(timesteps, data_dim),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=num_classes,
    unroll=True,
))
# output layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# optimizer
LR = 0.001
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train_bool,
          batch_size=1024, epochs=20,
          validation_data=(x_val, y_val_bool))
