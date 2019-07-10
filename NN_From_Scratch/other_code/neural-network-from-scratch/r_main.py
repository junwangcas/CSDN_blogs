import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

np.random.seed(0)
X, y = sklearn.datasets.make_moons(100, noise=0.01)

plt.scatter(X[:,0], X[:,1])
plt.title("origin data")
plt.show()

layer_dim = [2, 3, 2]



