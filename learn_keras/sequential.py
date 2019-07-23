import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn import datasets
import sklearn
from matplotlib import pyplot

# 生成虚拟数据
import numpy as np
sample_size = 100000
[x_total, y_total] = sklearn.datasets.make_moons(sample_size, shuffle = False)
x_training = x_total
y_training_int = np.zeros([len(y_total), 2])
for i in range(len(y_total)):
    if y_total[i] == 1:
        y_training_int[i, 1] = 1
    else:
        y_training_int[i, 0] = 1
y_training = (y_training_int == 1)
pyplot.scatter(x_training[0:int(sample_size/2),0],x_training[0:int(sample_size/2),1])
pyplot.scatter(x_training[int(sample_size/2+1):sample_size,0],x_training[int(sample_size/2+1):sample_size,1])
pyplot.show()
pyplot.title("original data")
# testing data
x_tesing = x_training
y_testing = y_training


model = Sequential()
# Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
# 在第一层必须指定所期望的输入数据尺寸：
# 在这里，是一个 20 维的向量。
model.add(Dense(128, activation='relu', input_dim=2))
#model.add(Dropout(0.5))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_training, y_training,
          epochs=20,
          batch_size=1280)
score = model.evaluate(x_tesing, y_testing, batch_size=128)
print(score)
