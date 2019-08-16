import pickle
import numpy as np
from keras.models import Sequential
from keras import layers


# 设置参数
# 是否reverse
is_reverse = True
#先制作一个数据集合
file_training_data = open("../calculator/cal_training.txt",'rb')
file_testing_data = open("../calculator/cal_testing.txt", 'rb')
training_samples = pickle.load(file_training_data)
testing_samples = pickle.load(file_testing_data)

num_traing = int(len(training_samples))
num_testing = int(len(training_samples))
X_train = np.zeros([num_traing, 8, 2])
y_train = np.zeros([num_traing, 8])

for i in range(num_traing):
    abit = np.unpackbits(np.uint8(training_samples[i][0]))
    bbit = np.unpackbits(np.uint8(training_samples[i][1]))
    cbit = np.transpose(np.unpackbits(np.uint8(training_samples[i][2])))
    if is_reverse:
        abit = abit[::-1]
        bbit = bbit[::-1]
        cbit = cbit[::-1]

    abbit = np.zeros((8,2),dtype=int)
    abbit[:,0] = abit
    abbit[:,1] = bbit
    X_train[i, :, :] = abbit
    y_train[i, :] = np.transpose(cbit)

print("data generation done")
RNN = layers.SimpleRNN
HIDDEN_SIZE = 16
BATCH_SIZE = 100
LAYERS = 1

print('Build model...')
model = Sequential()
MAXLEN = 8
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(8, 2)))
# As the decoder RNN's input, repeatedly provide with the last output of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
# model.add(layers.RepeatVector(1))
# The decoder RNN could be multiple layers stacked or a single layer.
#for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    #model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
#model.add(layers.TimeDistributed(layers.Dense(8, activation='softmax')))
model.add(layers.Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(X_train, y_train))
    # # Select 10 samples from the validation set at random so we can visualize
    # # errors.
    # for i in range(10):
    #     ind = np.random.randint(0, len(x_val))
    #     rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
    #     preds = model.predict_classes(rowx, verbose=0)
    #     q = ctable.decode(rowx[0])
    #     correct = ctable.decode(rowy[0])
    #     guess = ctable.decode(preds[0], calc_argmax=False)
    #     print('Q', q[::-1] if REVERSE else q, end=' ')
    #     print('T', correct, end=' ')
    #     if correct == guess:
    #         print(colors.ok + '☑' + colors.close, end=' ')
    #     else:
    #         print(colors.fail + '☒' + colors.close, end=' ')
    #     print(guess)

