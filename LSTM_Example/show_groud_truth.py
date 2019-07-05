
import numpy as np
import matplotlib.pyplot as plt
npzfile = np.load('ground_truth.npz')

a_ints = npzfile['arr_0']
a_s = npzfile['arr_1']
b_ints = npzfile['arr_2']
b_s = npzfile['arr_3']
c_ints = npzfile['arr_4']
c_s = npzfile['arr_5']

sample_size = a_ints.__len__()
# plt.scatter(range(sample_size), a_ints)
# plt.scatter(range(sample_size), b_ints)
# plt.scatter(range(sample_size), c_ints)
plt.scatter(range(sample_size), c_ints - a_ints - b_ints)

