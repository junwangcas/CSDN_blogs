from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt

# load data
npzfile = np.load('ground_truth.npz')
a_ints = npzfile['arr_0']
a_s = npzfile['arr_1']
b_ints = npzfile['arr_2']
b_s = npzfile['arr_3']
c_ints = npzfile['arr_4']
c_s = npzfile['arr_5']
sample_size = a_ints.__len__()
# generate training data
x_training = np.zeros((sample_size, 16))
y_training = np.zeros((sample_size, 8))
for i in range(sample_size):
    x_training[i, range(8)] = a_s[:, i]
    x_training[i, range(8, 16)] = b_s[:, i]
    y_training[i, range(8)] = c_s[:, i]


# define network
max_feature = 16
model = Sequential()
model.add(Embedding(max_feature, output_dim=8))
model.add(LSTM(16))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_training, y_training, batch_size=16, epochs=10)

