from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import SimpleRNN, Activation
from keras.optimizers import SGD
import numpy as np
import pickle
from get_cal_data import get_cal_data
from keras.optimizers import Adam
import matplotlib
#matplotlib.use('ggplot') #绘图风格设置
import matplotlib.pyplot as plt

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
    output_dim=1024,
    unroll=True,
))
# output layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# optimizer
LR = 0.001
adam = Adam(LR)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model_trained = model.fit(x_train, y_train_bool,
          batch_size=1024, epochs=20,
          validation_data=(x_val, y_val_bool))

## plot;
# Plot accuracy
plt.subplot(211)
plt.plot(model_trained.history['acc'])
plt.plot(model_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(212)
plt.plot(model_trained.history['loss'])
plt.plot(model_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()

plt.tight_layout(h_pad=1.0)
plt.savefig('./history-graph.png')

print(str(model_trained.history['val_acc'][-1])[:6] +
      "(max: " + str(max(model_trained.history['val_acc']))[:6] + ")")
print("Done.")