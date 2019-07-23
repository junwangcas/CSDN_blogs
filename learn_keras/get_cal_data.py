import numpy as np
import pickle
def get_cal_data(data_dim = 2, timesteps = 8, num_classes = 256):
    # load data
    is_reverse = True
    # 先制作一个数据集合
    file_training_data = open("cal_training.txt", 'rb')
    file_testing_data = open("cal_testing.txt", 'rb')
    training_samples = pickle.load(file_training_data)
    testing_samples = pickle.load(file_testing_data)

    x_train = np.random.random((len(training_samples), timesteps, data_dim))
    y_train = np.zeros([len(training_samples), num_classes])
    x_val = np.random.random((len(testing_samples), timesteps, data_dim))
    y_val = np.zeros([len(testing_samples), num_classes])

    for i in range(len(training_samples)):
        abit = np.unpackbits(np.uint8(training_samples[i][0]))
        bbit = np.unpackbits(np.uint8(training_samples[i][1]))
        cbit = np.transpose(np.unpackbits(np.uint8(training_samples[i][2])))
        if is_reverse:
            abit = abit[::-1]
            bbit = bbit[::-1]
            cbit = cbit[::-1]

        abbit = np.zeros((8, 2), dtype=int)
        abbit[:, 0] = abit
        abbit[:, 1] = bbit
        x_train[i, :, 0] = abit
        x_train[i, :, 1] = bbit
        y_train[i, np.uint8(training_samples[i][2])] = 1
        #y_train[i,:] = cbit

    y_train_bool = (y_train == 1)

    # check the training data is correct.
    # for i in range(len(x_train)):
    #     abit_reverse = x_train[i, :, 0]
    #     bbit_reverse = x_train[i, :, 1]
    #     c = np.argmax(y_train[i, :])
    #     abit = abit_reverse[::-1]
    #     bbit = bbit_reverse[::-1]
    #     print("training data")
    #     print(np.packbits(abit==1))
    #     print(np.packbits(bbit==1))
    #     print(c)


    for i in range(len(testing_samples)):
        abit = np.unpackbits(np.uint8(testing_samples[i][0]))
        bbit = np.unpackbits(np.uint8(testing_samples[i][1]))
        cbit = np.transpose(np.unpackbits(np.uint8(testing_samples[i][2])))
        if is_reverse:
            abit = abit[::-1]
            bbit = bbit[::-1]
            cbit = cbit[::-1]

        abbit = np.zeros((8, 2), dtype=int)
        abbit[:, 0] = abit
        abbit[:, 1] = bbit
        x_val[i, :, 0] = abit
        x_val[i, :, 1] = bbit
        y_val[i, np.uint8(testing_samples[i][2])] = 1
        #y_val[i, :] = cbit

    y_val_bool = (y_val == 1)

    return x_train, y_train, x_val, y_val

def get_cal_data_one_dimension(data_dim = 16, num_classes = 256):
    # load data
    is_reverse = True
    # 先制作一个数据集合
    file_training_data = open("cal_training.txt", 'rb')
    file_testing_data = open("cal_testing.txt", 'rb')
    training_samples = pickle.load(file_training_data)
    testing_samples = pickle.load(file_testing_data)

    x_train = np.random.random((len(training_samples), data_dim))
    y_train = np.zeros([len(training_samples), num_classes])
    x_val = np.random.random((len(testing_samples), data_dim))
    y_val = np.zeros([len(testing_samples), num_classes])

    for i in range(len(training_samples)):
        abit = np.unpackbits(np.uint8(training_samples[i][0]))
        bbit = np.unpackbits(np.uint8(training_samples[i][1]))
        cbit = np.transpose(np.unpackbits(np.uint8(training_samples[i][2])))
        if is_reverse:
            abit = abit[::-1]
            bbit = bbit[::-1]
            cbit = cbit[::-1]

        x_train[i,0:8] = abit
        x_train[i,8:16] = bbit
        y_train[i, np.uint8(training_samples[i][2])] = 1

    y_train_bool = (y_train == 1)

    for i in range(len(testing_samples)):
        abit = np.unpackbits(np.uint8(testing_samples[i][0]))
        bbit = np.unpackbits(np.uint8(testing_samples[i][1]))
        cbit = np.transpose(np.unpackbits(np.uint8(testing_samples[i][2])))
        if is_reverse:
            abit = abit[::-1]
            bbit = bbit[::-1]
            cbit = cbit[::-1]

        x_val[i, 0:8] = abit
        x_val[i, 8:16] = bbit
        y_val[i, np.uint8(testing_samples[i][2])] = 1

    y_val_bool = (y_val == 1)

    return x_train, y_train_bool, x_val, y_val_bool