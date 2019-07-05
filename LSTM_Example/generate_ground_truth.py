import numpy as np

np.random.seed(0)

# settings;
sample_size = 100000

# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

a_ints = np.zeros(sample_size)
a_s = np.zeros((binary_dim, sample_size))
b_ints = np.zeros(sample_size)
b_s = np.zeros((binary_dim, sample_size))
c_ints = np.zeros(sample_size)
c_s = np.zeros((binary_dim, sample_size))


for j in range(sample_size):
    a_int = np.random.randint(largest_number / 2)
    a = int2binary[a_int]

    b_int = np.random.randint(largest_number / 2)
    b = int2binary[b_int]

    c_int = a_int + b_int
    c = int2binary[c_int]

    a_ints[j] = a_int
    a_s[:, j] = a
    b_ints[j] = b_int
    b_s[:, j] = b
    c_ints[j] = c_int
    c_s[:, j] = c

# save these data
np.savez('ground_truth', a_ints, a_s, b_ints, b_s, c_ints, c_s)



