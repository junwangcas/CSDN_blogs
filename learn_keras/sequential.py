import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn import datasets
import sklearn
from matplotlib import pyplot

# 生成虚拟数据
import numpy as np
sample_size = 10000
[x_total, y_total] = sklearn.datasets.make_moons(sample_size, shuffle = False)
x_training = x_total
y_training = y_total
pyplot.scatter(x_training[0:int(sample_size/2),0],x_training[0:int(sample_size/2),1])
pyplot.scatter(x_training[int(sample_size/2+1):sample_size,0],x_training[int(sample_size/2+1):sample_size,1])
pyplot.show()
pyplot.title("original data")




x_train = np.random.random((1000, 2))
y_train = keras.utils.to_categorical(np.random.randint(1, size=(1000, 1)), num_classes=2)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
# 在第一层必须指定所期望的输入数据尺寸：
# 在这里，是一个 20 维的向量。
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
