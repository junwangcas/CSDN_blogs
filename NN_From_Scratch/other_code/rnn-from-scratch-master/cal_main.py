from CalRNN import Model
import pickle
import numpy as np
cal_rnn = Model()

#先制作一个数据集合
file_training_data = open("calculator/cal_training.txt",'rb')
training_samples = pickle.load(file_training_data)
abit = np.unpackbits(np.uint8(training_samples[0][0]))
bbit = np.unpackbits(np.uint8(training_samples[0][1]))
cbit = np.unpackbits(np.uint8(training_samples[0][2]))

X_train = np.zeros((2,8),dtype=int)
y_train = np.zeros((1,8),dtype=int)
X_train[0,:] = abit
X_train[1,:] = bbit
y_train = cbit
losses = cal_rnn.train(X_train, y_train, learning_rate=0.005, nepoch=10, evaluate_loss_after=1)
