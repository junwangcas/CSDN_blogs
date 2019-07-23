import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn import datasets
import sklearn
from matplotlib import pyplot
from get_cal_data import get_cal_data_one_dimension

# 生成虚拟数据
import numpy as np

[x_training, y_training, x_tesing, y_testing] = get_cal_data_one_dimension()


model = Sequential()
# Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
# 在第一层必须指定所期望的输入数据尺寸：
# 在这里，是一个 20 维的向量。
model.add(Dense(1024, activation='relu', input_dim=16))
#model.add(Dropout(0.5))
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(256, activation='softmax'))

sgd = SGD(lr=1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_training, y_training,
          epochs=2,
          batch_size=512)
print(model.predict(x_tesing[0,:].reshape(1,16)))

score = model.evaluate(x_tesing, y_testing, batch_size=128)
print(score)
