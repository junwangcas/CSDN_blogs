import numpy
from matplotlib import pyplot as plt

x = numpy.arange(10)
y = numpy.array([5,3,4,2,7,5,4,6,3,2])
for i in range(10):
    plt.subplot(2,1,1)
    plt.show(block=False)
    print(i)
    plt.pause(0.1)
