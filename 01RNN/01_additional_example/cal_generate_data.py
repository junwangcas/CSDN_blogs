import numpy as np
import pickle

bit_length = 8
largest_number = pow(2, 8)
file_training_data = open("cal_training.txt",'wb')
file_int2bits_data = open("cal_int2bits.txt",'wb')
file_testing_data = open("cal_testing.txt", 'wb')

int2bits = dict()
for i in range(largest_number):
    int_one = i
    int_bits = np.unpackbits(np.uint8(int_one))
    int2bits[i] = int_bits

training_samples = list()
training_num = 20000
for i in range(training_num):
    a_int = np.random.randint(low=0, high = largest_number/2)
    b_int = np.random.randint(low=0, high = largest_number/2)
    c_int = a_int + b_int
    sample = [a_int, b_int, c_int]
    training_samples.append(sample)

testing_samples = list()
testing_num = 1000
for i in range(testing_num):
    a_int = np.random.randint(low=0, high=largest_number / 2)
    b_int = np.random.randint(low=0, high=largest_number / 2)
    c_int = a_int + b_int
    sample = [a_int, b_int, c_int]
    testing_samples.append(sample)

pickle.dump(training_samples, file_training_data, 0)
pickle.dump(int2bits, file_int2bits_data, 0)
pickle.dump(testing_samples, file_testing_data, 0)
file_training_data.close()
file_int2bits_data.close()
file_testing_data.close()
