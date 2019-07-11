import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
file_w = open("data/w.txt",'rb')
file_b = open("data/b.txt",'rb')
file_forward = open("data/forward.txt",'rb')
## read log
ws = pickle.load(file_w)
bs = pickle.load(file_b)
forwards = pickle.load(file_forward)
## visualize
# fig, ax = plt.subplots()
# ws_1 = ws[0][0]
# ax.matshow(ws_1, cmap = cm.Greys_r)
# for (i, j), z in np.ndenumerate(ws_1):
#     ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
#             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
# plt.show()


for i in range(len(forwards)):
    # one pass
    forword = forwards[i]
    # input layer
    fig = plt.subplot(3,3,1)
    plt.suptitle(["pass ", str(i)])
    plt.title("input layer")
    a_1 = forword[0][2] # (200, 2)
    x = range(len(a_1[:,0]))
    plt.plot(x, a_1[:,0])
    plt.subplot(3,3,4)
    plt.plot(x, a_1[:,1])
    # hidden layer
    plt.subplot(3,3,2)
    plt.title("hidden layer")
    a_2 = forword[1][2]
    x = range(len(a_2[:, 0]))
    plt.plot(x, a_2[:, 0])
    plt.subplot(3,3,5)
    plt.plot(x, a_2[:, 1])
    plt.subplot(3, 3, 8)
    plt.plot(x, a_2[:, 2])
    # output layer
    plt.subplot(3, 3, 3)
    plt.title("output layer")
    a_3 = forword[2][2]
    x = range(len(a_3[:, 0]))
    plt.plot(x, a_3[:, 0])
    plt.subplot(3, 3, 6)
    plt.plot(x, a_3[:, 1])
    plt.show(block=False)
    plt.pause(0.0001)
    plt.clf()
    #plt.waitforbuttonpress(timeout=-1)

# show forward layers
for i in range(len(forwards)):
    # one pass
    forword = forwards[i]
    # input layer
    fig = plt.subplot(3,3,1)
    plt.suptitle(["pass ", str(i)])
    plt.title("input layer")
    a_1 = forword[0][2] # (200, 2)
    plt.hist(x=a_1[:,0])
    plt.subplot(3,3,4)
    plt.hist(x=a_1[:,1])
    # hidden layer
    plt.subplot(3,3,2)
    plt.title("hidden layer")
    a_2 = forword[1][2]
    plt.hist(x=a_2[:,0])
    plt.subplot(3,3,5)
    plt.hist(x=a_2[:, 1])
    plt.subplot(3, 3, 8)
    plt.hist(x=a_2[:, 2])
    # output layer
    plt.subplot(3, 3, 3)
    plt.title("output layer")
    a_3 = forword[2][2]
    plt.hist(x=a_3[:, 0])
    plt.subplot(3, 3, 6)
    plt.hist(x=a_3[:, 1])
    plt.show(block=False)
    plt.pause(0.1)
    plt.clf()
    #plt.waitforbuttonpress(timeout=0.1)

